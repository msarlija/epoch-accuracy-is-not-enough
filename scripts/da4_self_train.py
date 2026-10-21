# -*- coding: utf-8 -*-
"""DA-4: confidence-weighted mean-teacher self-training of the classifier head.

DA-1 to DA-3 (scripts/domain_adapt.py) are inference-only: a diagonal affine
fitted by moment matching at the head's 64-d hidden layer. DA-4 keeps every
weight upstream of the head frozen and instead FITS the head itself
(``dense`` 391->64 and ``fc`` 64->5, 25,413 parameters) on the unlabelled
target cohort with the recipe of MC-PanDA (Martinovic, Saric, Segvic, ECCV
2024), transposed from panoptic segmentation to a hypnogram:

    pixel  -> epoch
    mask   -> bout, a maximal run of same-stage epochs in the TEACHER's hypnogram
    image  -> 60-epoch window

* student = the fold's head, trained; teacher = EMA copy, never trained.
* teacher sees clean features, student sees perturbed ones (feature noise and
  dropout, the analogue of colour jitter and blur).
* bout-wide loss scaling: each pseudo-labelled epoch's loss is multiplied by
  the fraction of its bout whose teacher confidence exceeds tau1
  (``--weighting bout``). ``window`` is the image-wide scaling of EDAPS/
  DAFormer, ``none`` is plain consistency.
* confidence-based filtering: epochs whose teacher confidence is below tau2
  receive no gradient at all.
* a supervised cross-entropy on the fold's internal TRAIN split anchors the
  head to the source throughout (the L_s term).

Everything runs on cached head inputs: the CNN, aggregation, time branch,
fusion and GRUs are run once per recording and never again, so one
adaptation is seconds and the whole 9-cohort inductive protocol is minutes.

Protocol ``halves``: within each cohort the SUBJECTS are split in two; the
head adapted on half A scores half B and vice versa, so every recording is
scored by a head that never saw it (inductive, the analogue of LOSO in
domain_adapt.py). ``transductive`` adapts on the whole cohort and scores it.
``pooled`` adapts on the A halves of ALL cohorts at once.

Controls, run under the same protocol and step budget:
  source-only   lambda_t = 0: separates "more head training" from adaptation
  oracle        target CE with TRUE labels: the supervised ceiling of this
                parameterisation (not deployable)

Prior correction (§7.3 of docs/methods.md) is applied to every condition and
reported beside the raw figures.

    python scripts/da4_self_train.py --fold 0 --cohorts ANPHY DODH DREAMS-S \
        --tau1 0.8 0.9 0.95 --tau2 0.5 0.7 0.8 --weighting none window bout
    python scripts/da4_self_train.py --fold 0 --tau1 0.9 --tau2 0.7 \
        --weighting bout --conditions da4 source-only oracle --tag main
"""

import argparse
import copy
import itertools
import json
import os
import sys
import time
import zlib
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config                            # noqa: E402
from data import manifest as mf                           # noqa: E402
from data.sampling import apply_cohort_caps               # noqa: E402
from eval.stats import paired_bootstrap                   # noqa: E402
from models.model import build_model, reconcile_head_time_skip  # noqa: E402
from scripts.run_experiment import write_metrics          # noqa: E402
from scripts.domain_adapt import collect_features         # noqa: E402

#: trailing raw time harmonics in the 391-d head input; never perturbed
TIME_DIMS = 7
WINDOW = 60
N_CLASSES = 5
STAGE_NAMES = ("W", "N1", "N2", "N3", "REM")


# ------------------------------------------------------------------ features
def cached_features(model, df, cfg, device, trim_wake, cache_dir, label):
    """{rec_id: (X (T, 391) f32, proba0 (T, 5), y (T,))}, cached per recording.

    The head input is a deterministic function of the frozen network and the
    recording, so it is computed once per fold and reused by every condition.
    """
    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    missing = []
    for rid in df.rec_id:
        p = os.path.join(cache_dir, f"{rid}.npz")
        if os.path.exists(p):
            z = np.load(p)
            out[rid] = (z["x"], z["proba"], z["y"])
        else:
            missing.append(rid)
    if missing:
        t0 = time.time()
        print(f"  {label}: extracting {len(missing)} recordings "
              f"({len(df) - len(missing)} cached)", flush=True)
        sub = df[df.rec_id.isin(missing)].reset_index(drop=True)
        feats = collect_features(model, sub, cfg, device, trim_wake)
        for rid, (x, _pred, proba, y, _agg, _tso) in feats.items():
            np.savez(os.path.join(cache_dir, f"{rid}.npz"),
                     x=x.astype(np.float32), proba=proba.astype(np.float32),
                     y=y.astype(np.int8))
            out[rid] = (x.astype(np.float32), proba.astype(np.float32),
                        y.astype(np.int8))
        print(f"  {label}: done in {time.time() - t0:.0f} s", flush=True)
    return out


def subject_halves(df, seed):
    """{cohort: (set of rec_ids in half A, set in half B)}, split by SUBJECT."""
    halves = {}
    for c, g in df.groupby("cohort"):
        subs = np.asarray(sorted(g.subject_id.astype(str).unique()))
        rng = np.random.default_rng((zlib.crc32(c.encode()) + seed) % 2 ** 32)
        rng.shuffle(subs)
        a = set(subs[: len(subs) // 2])
        ra = set(g[g.subject_id.astype(str).isin(a)].rec_id)
        rb = set(g.rec_id) - ra
        halves[c] = (ra, rb)
    return halves


# ---------------------------------------------------------------- diagnostic
def confidence_diagnostic(tgt, tgt_df, taus):
    """Precision and coverage of the UNADAPTED model's confident epochs.

    This is the quantity self-training lives on: a pseudo-label is only useful
    where the teacher is confident AND right. Uses the labels, so it is a
    design-time check, not part of the method.
    """
    rows = []
    for c, g in tgt_df.groupby("cohort"):
        P = np.concatenate([tgt[r][1] for r in g.rec_id])
        y = np.concatenate([tgt[r][2] for r in g.rec_id]).astype(np.int64)
        ok = y >= 0
        P, y = P[ok], y[ok]
        conf, yhat = P.max(1), P.argmax(1)
        err_total = float((yhat != y).sum())
        for tau in taus:
            keep = conf >= tau
            row = {"cohort": c, "tau": tau, "coverage": keep.mean(),
                   "precision": (yhat[keep] == y[keep]).mean() if keep.any() else np.nan,
                   "share_of_error_kept": ((yhat != y) & keep).sum() / max(err_total, 1)}
            for k, name in enumerate(STAGE_NAMES):
                m = keep & (yhat == k)
                row[f"prec_{name}"] = (y[m] == k).mean() if m.any() else np.nan
                row[f"cov_{name}"] = m.sum() / max((yhat == k).sum(), 1)
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ weights
def pseudo_weights(yhat, conf, rec_bounds, tau1, weighting):
    """Per-epoch loss weight from the TEACHER's hypnogram.

    ``bout``: lambda of the maximal same-stage run containing the epoch, as
    the fraction of its epochs with confidence > tau1 (MC-PanDA's mask-wide
    scaling). ``window``: the same over fixed 60-epoch windows (image-wide
    scaling). ``none``: all ones.
    """
    n = len(yhat)
    if weighting == "none":
        return np.ones(n, dtype=np.float32)
    base, _, norm = weighting.partition("-")
    seg = np.zeros(n, dtype=np.int64)
    next_id = 0
    for lo, hi in rec_bounds:
        if base == "bout":
            yy = yhat[lo:hi]
            change = np.r_[1, (yy[1:] != yy[:-1]).astype(np.int64)]
            ids = np.cumsum(change) - 1
        else:
            ids = np.arange(hi - lo) // WINDOW
        seg[lo:hi] = ids + next_id
        next_id += int(ids.max()) + 1
    hit = (conf > tau1).astype(np.float64)
    num = np.bincount(seg, weights=hit, minlength=next_id)
    den = np.bincount(seg, minlength=next_id).astype(np.float64)
    w = (num / np.maximum(den, 1))[seg]
    if norm == "norm":
        # Rescale to mean 1 WITHIN each predicted stage. Raw bout weights are
        # dominated by bout length and stage prevalence (long confident W and
        # N2 runs score near 1, short N1 and transitional REM runs near 0), so
        # the unnormalised loss drifts toward the majority stages. Normalising
        # keeps the within-stage ranking of bouts and removes the between-stage
        # imbalance.
        for k in range(N_CLASSES):
            m = yhat == k
            if m.any() and w[m].mean() > 0:
                w[m] = w[m] / w[m].mean()
    return w.astype(np.float32)


def class_balanced_keep(conf, yhat, tau2, floor_q):
    """Keep epochs with conf >= tau2, but never drop more than (1 - floor_q)
    of any predicted class: the per-class threshold is min(tau2, the floor_q
    quantile of that class's confidence). Guards N1, whose confidence is
    structurally lowest."""
    keep = np.zeros(len(conf), dtype=bool)
    for k in range(N_CLASSES):
        m = yhat == k
        if not m.any():
            continue
        thr = min(tau2, float(np.quantile(conf[m], floor_q)))
        keep[m] = conf[m] >= thr
    return keep


# ------------------------------------------------------------------ training
def parse_tau2(text):
    """'0.9' -> 0.9; '0.9,0.6,0.9,0.8,0.9' -> tuple of five floats."""
    if "," in str(text):
        vals = tuple(float(v) for v in str(text).split(","))
        if len(vals) != N_CLASSES:
            raise ValueError(f"--tau2 needs {N_CLASSES} values, got {text!r}")
        return vals
    return float(text)


def tau2_vector(tau2):
    return np.asarray(tau2 if isinstance(tau2, tuple) else [tau2] * N_CLASSES,
                      dtype=np.float32)


def tau2_label(tau2):
    if isinstance(tau2, tuple):
        return "-".join(f"{v:g}" for v in tau2)
    return f"{tau2:g}"


class HP:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def tag(self):
        return (f"t1{self.tau1:g}_t2{tau2_label(self.tau2)}_l{self.lam:g}_s{self.steps}"
                f"_{self.weighting}_e{self.ema:g}_n{self.noise:g}_d{self.drop:g}"
                f"_lr{self.lr:g}"
                + (f"_cb{self.cb_floor:g}" if self.cb_floor > 0 else ""))


def head_forward(head, x):
    """Logits from the head (its ActivationAdapters are identity here)."""
    return head(x)


def adapt_head(head0, src, val, tgt_X, tgt_y, rec_bounds, hp, mode, device,
               seed):
    """Return (adapted head, diagnostics dict).

    src = (X, y) tensors on device, val = (X, y) for the drift guard, tgt_X
    (N, 391) on device, tgt_y the TRUE labels (used by the oracle and for
    pseudo-label diagnostics only).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    student = copy.deepcopy(head0).to(device)
    teacher = copy.deepcopy(head0).to(device)
    for p in list(student.parameters()) + list(teacher.parameters()):
        p.requires_grad_(False)
    params = list(student.dense.parameters()) + list(student.fc.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=hp.lr)
    src_X, src_y = src
    N_s, N_t = len(src_X), len(tgt_X)
    feat_sd = src_X[:, :-TIME_DIMS].std(0, keepdim=True)

    def perturb(x):
        if hp.noise <= 0 and hp.drop <= 0:
            return x
        core, tm = x[:, :-TIME_DIMS], x[:, -TIME_DIMS:]
        if hp.noise > 0:
            core = core + hp.noise * feat_sd * torch.randn_like(core)
        if hp.drop > 0:
            core = F.dropout(core, hp.drop, training=True)
        return torch.cat([core, tm], 1)

    yhat = conf = None
    w = keep = None
    diag = {"pl_precision_start": np.nan, "pl_precision_end": np.nan,
            "keep_frac": np.nan, "mean_w": np.nan}
    tgt_y_np = tgt_y.cpu().numpy() if tgt_y is not None else None

    def refresh():
        with torch.no_grad():
            P = torch.softmax(head_forward(teacher, tgt_X), 1)
            cf, yh = P.max(1)
        cf, yh = cf.cpu().numpy(), yh.cpu().numpy()
        ww = pseudo_weights(yh, cf, rec_bounds, hp.tau1, hp.weighting)
        if hp.cb_floor > 0:
            kp = class_balanced_keep(cf, yh, float(np.max(tau2_vector(hp.tau2))),
                                     hp.cb_floor)
        else:
            kp = cf >= tau2_vector(hp.tau2)[yh]
        return (torch.from_numpy(yh).to(device), cf,
                torch.from_numpy(ww * kp).to(device), kp)

    pool = None
    for step in range(hp.steps):
        if mode == "da4" and step % hp.refresh == 0:
            yhat, conf, w, keep = refresh()
            pool = torch.nonzero(torch.from_numpy(keep)).squeeze(1)
            if step == 0 and tgt_y_np is not None:
                ok = keep & (tgt_y_np >= 0)
                diag["pl_precision_start"] = float(
                    (yhat.cpu().numpy()[ok] == tgt_y_np[ok]).mean()) if ok.any() else np.nan
                diag["keep_frac"] = float(keep.mean())
                yh_np = yhat.cpu().numpy()
                for k, name in enumerate(STAGE_NAMES):
                    m = yh_np == k
                    diag[f"keep_{name}"] = float(keep[m].mean()) if m.any() else np.nan
                diag["mean_w"] = (float(w[torch.from_numpy(keep).to(device)].mean())
                                  if keep.any() else 0.0)
        si = torch.randint(0, N_s, (hp.batch,), generator=g).to(device)
        loss = F.cross_entropy(head_forward(student, perturb(src_X[si])),
                               src_y[si], ignore_index=-1)
        if mode == "da4" and len(pool) > 0:
            ti = pool[torch.randint(0, len(pool), (hp.batch,), generator=g)].to(device)
            ce = F.cross_entropy(head_forward(student, perturb(tgt_X[ti])),
                                 yhat[ti], reduction="none")
            loss = loss + hp.lam * (ce * w[ti]).sum() / hp.batch
        elif mode == "oracle":
            ti = torch.randint(0, N_t, (hp.batch,), generator=g).to(device)
            loss = loss + hp.lam * F.cross_entropy(
                head_forward(student, perturb(tgt_X[ti])), tgt_y[ti],
                ignore_index=-1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if hp.ema < 1.0:
            with torch.no_grad():
                for pt, ps in zip(teacher.parameters(), student.parameters()):
                    pt.mul_(hp.ema).add_(ps, alpha=1.0 - hp.ema)
    if mode == "da4" and tgt_y_np is not None:
        yhat, conf, w, keep = refresh()
        ok = keep & (tgt_y_np >= 0)
        diag["pl_precision_end"] = float(
            (yhat.cpu().numpy()[ok] == tgt_y_np[ok]).mean()) if ok.any() else np.nan
    with torch.no_grad():
        vX, vy = val
        acc0 = (head_forward(head0, vX).argmax(1) == vy)[vy >= 0].float().mean()
        acc1 = (head_forward(student, vX).argmax(1) == vy)[vy >= 0].float().mean()
    diag["src_val_acc_base"] = float(acc0)
    diag["src_val_acc"] = float(acc1)
    return student.eval(), diag


# ------------------------------------------------------------------ scoring
def score(head, X, device):
    with torch.no_grad():
        lg = head_forward(head, torch.from_numpy(X).to(device))
        return torch.softmax(lg, 1).cpu().numpy()


def make_record(row, y_true, proba):
    pred = proba.argmax(1)
    return {"rec_id": row["rec_id"], "cohort": row["cohort"],
            "subject_id": row.get("subject_id", row["rec_id"]),
            "montage_family": row["montage_family"],
            "n_channels": row["n_channels"], "y_true": y_true,
            "y_pred": np.where(y_true == -1, -1, pred), "proba": proba}


def prior_correct(recs, base_recs):
    """Restore the predicted class marginal of the unadapted model, per cohort,
    from the model's own outputs (label-free, §7.3)."""
    by_cohort = defaultdict(lambda: [[], []])
    for ra, rb in zip(recs, base_recs):
        by_cohort[ra["cohort"]][0].append(ra["proba"])
        by_cohort[ra["cohort"]][1].append(rb["proba"])
    adjust = {c: np.log(np.maximum(np.concatenate(pb).mean(0), 1e-8) /
                        np.maximum(np.concatenate(pa).mean(0), 1e-8))
              for c, (pa, pb) in by_cohort.items()}
    out = []
    for ra in recs:
        lp = np.log(np.maximum(ra["proba"], 1e-12)) + adjust[ra["cohort"]]
        lp -= lp.max(1, keepdims=True)
        pr = np.exp(lp)
        pr /= pr.sum(1, keepdims=True)
        yt = ra["y_true"]
        out.append({**ra, "proba": pr, "y_pred": np.where(yt == -1, -1, pr.argmax(1))})
    return out


def summarise(A, B, label):
    """Paired, subject-clustered bootstrap of adapted (A) minus base (B)."""
    row = {"condition": label, "n_rec": len(A)}
    for m in ("kappa", "accuracy", "f1_macro", "rol_ae_min"):
        r = paired_bootstrap(A, B, m, n_boot=5000, seed=0)
        row[f"{m}_base"] = B[m].mean()
        row[f"{m}_delta"] = r["mean_diff"]
        row[f"{m}_p"] = r["p_value"]
    # within-10-min REM-onset agreement (paper's primary endpoint family)
    a10 = (A.rol_ae_min <= 10).mean()
    b10 = (B.rol_ae_min <= 10).mean()
    row["within10_base"] = b10
    row["within10_delta"] = a10 - b10
    for k in STAGE_NAMES:
        row[f"f1_{k}_delta"] = A[f"f1_{k}"].mean() - B[f"f1_{k}"].mean()
    return row


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default=os.path.expanduser(os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging") + "/runs"))
    ap.add_argument("--prefix", default="final")
    ap.add_argument("--config-prefix", default="final")
    ap.add_argument("--arm", default="full")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cohorts", nargs="*", default=None,
                    help="external cohorts to run (default: all in the config)")
    ap.add_argument("--protocol", default="halves",
                    choices=("halves", "transductive", "pooled"))
    ap.add_argument("--adapt-k", nargs="*", type=int, default=[],
                    help="protocol=halves: cap the adaptation half at K "
                         "subjects (drawn deterministically per cohort and "
                         "half), one condition per value. With the oracle "
                         "condition this is the labelled K-curve: how many "
                         "annotated subjects a site needs.")
    ap.add_argument("--conditions", nargs="+", default=["da4"],
                    choices=("da4", "source-only", "oracle"))
    # grid dimensions (every combination is run from the one feature cache)
    ap.add_argument("--tau1", nargs="+", type=float, default=[0.9],
                    help="bout/window scaling threshold")
    ap.add_argument("--tau2", nargs="+", default=["0.7"],
                    help="epoch filtering threshold: a single value, or five "
                         "comma-separated per-stage values in the order "
                         "W,N1,N2,N3,REM (e.g. 0.9,0.6,0.9,0.8,0.9). A stage "
                         "with weak feature evidence such as N1 never reaches "
                         "the confidence of W or N2, so one global threshold "
                         "silently removes it from the target loss.")
    ap.add_argument("--lam", nargs="+", type=float, default=[1.0],
                    help="weight of the target loss against the source CE")
    ap.add_argument("--steps", nargs="+", type=int, default=[500])
    ap.add_argument("--weighting", nargs="+", default=["bout"],
                    choices=("none", "window", "bout", "window-norm",
                             "bout-norm"),
                    help="-norm variants rescale the weights to mean 1 within "
                         "each predicted stage")
    ap.add_argument("--ema", nargs="+", type=float, default=[0.99],
                    help="teacher EMA rate per step; 1.0 freezes the teacher "
                         "at the source head")
    ap.add_argument("--noise", nargs="+", type=float, default=[0.1],
                    help="student feature noise, in units of the per-dim "
                         "source std")
    ap.add_argument("--drop", nargs="+", type=float, default=[0.1],
                    help="student feature dropout")
    ap.add_argument("--cb-floor", nargs="+", type=float, default=[0.0],
                    help=">0 enables class-balanced filtering: keep at least "
                         "this fraction of every predicted class")
    ap.add_argument("--lr", nargs="+", type=float, default=[1e-4])
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--refresh", type=int, default=25,
                    help="recompute teacher pseudo-labels every N steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("--out-root", default=os.path.expanduser(os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging") + "/analysis/da4"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.tau2 = [parse_tau2(t) for t in args.tau2]

    device = args.device
    fold = args.fold
    run_dir = f"{args.runs_root}/{args.prefix}_{args.arm}_seed0_fold{fold}of10"
    cfg = load_config(f"configs/{args.config_prefix}_{args.arm}.yaml")
    eval_cfg = cfg.get("eval", {})
    manifest = mf.load_manifest(cfg["data_dir"])
    if cfg.get("require_qc_pass", True):
        manifest = manifest[manifest["qc_pass"] == 1].reset_index(drop=True)
    with open(os.path.join(run_dir, "split_manifest.json")) as fh:
        split_ids = json.load(fh)["splits"]
    src_df = manifest[manifest.rec_id.isin(split_ids["train"])].reset_index(drop=True)
    val_df = manifest[manifest.rec_id.isin(split_ids["val"])].reset_index(drop=True)
    tgt_df = manifest[manifest.cohort.isin(cfg["external_cohorts"])]
    caps = cfg.get("cohort_caps") or {}
    if caps:
        tgt_df = apply_cohort_caps(tgt_df, caps, seed=cfg.get("seed", 0))
    tgt_df = tgt_df.reset_index(drop=True)
    if args.cohorts:
        tgt_df = tgt_df[tgt_df.cohort.isin(args.cohorts)].reset_index(drop=True)

    ckpt = torch.load(os.path.join(run_dir, "stage_b.pt"), map_location="cpu")
    mcfg = reconcile_head_time_skip(dict(load_config(
        os.path.join(run_dir, "resolved_config.yaml"))["model"]),
        ckpt["state_dict"], os.path.basename(run_dir))
    model = build_model(mcfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    head0 = copy.deepcopy(model.seq_head).to(device).eval()
    n_trainable = sum(p.numel() for p in head0.dense.parameters()) + \
        sum(p.numel() for p in head0.fc.parameters())
    trim = cfg.get("trim_wake") or {}
    trim_wake = ((float(trim.get("min_minutes", 20.0)),
                  float(trim.get("max_minutes", 40.0)))
                 if trim.get("enabled") else None)

    cache = os.path.join(args.out_root, "features", f"fold{fold}")
    print(f"fold {fold}: {len(src_df)} source (train), {len(val_df)} val, "
          f"{len(tgt_df)} target recordings in {tgt_df.cohort.nunique()} cohorts; "
          f"{n_trainable} trainable head parameters", flush=True)
    src = cached_features(model, src_df, cfg, device, trim_wake,
                          os.path.join(cache, "internal_trimmed"), "source")
    val = cached_features(model, val_df, cfg, device, trim_wake,
                          os.path.join(cache, "internal_trimmed"), "val")
    tgt = cached_features(model, tgt_df, cfg, device, None,
                          os.path.join(cache, "external"), "target")
    del model
    torch.cuda.empty_cache()

    out_dir = os.path.join(args.out_root, f"fold{fold}_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)

    # ---- design-time diagnostic of the unadapted model's confidence
    scalar_t2 = {t for t in args.tau2 if not isinstance(t, tuple)}
    diag = confidence_diagnostic(tgt, tgt_df, sorted(set(args.tau1) | scalar_t2
                                                      | {0.5, 0.8, 0.9, 0.95}))
    diag.to_csv(os.path.join(out_dir, "confidence_diagnostic.csv"), index=False)
    print("\nconfidence diagnostic (unadapted model; precision of kept pseudo-labels)")
    piv = diag.pivot(index="cohort", columns="tau", values=["coverage", "precision"])
    print(piv.round(3).to_string())

    def to_dev(d, keys):
        X = np.concatenate([d[k][0] for k in keys])
        y = np.concatenate([d[k][2] for k in keys]).astype(np.int64)
        return (torch.from_numpy(X).to(device), torch.from_numpy(y).to(device))

    src_t = to_dev(src, list(src_df.rec_id))
    val_t = to_dev(val, list(val_df.rec_id))
    print(f"source epochs on device: {len(src_t[0])}", flush=True)

    # ---- adaptation groups: (train rec_ids, eval rec_ids) pairs
    halves = subject_halves(tgt_df, args.seed)
    subj_of = dict(zip(tgt_df.rec_id, tgt_df.subject_id.astype(str)))

    def cap_subjects(ids, k, salt):
        if not k:
            return sorted(ids)
        subs = np.asarray(sorted({subj_of[r] for r in ids}))
        rng = np.random.default_rng((zlib.crc32(salt.encode()) + args.seed) % 2 ** 32)
        rng.shuffle(subs)
        keep = set(subs[:k])
        return sorted(r for r in ids if subj_of[r] in keep)

    def build_groups(k=0):
        groups = []     # (name, cohort or 'pooled', adapt ids, eval ids)
        if args.protocol == "halves":
            for c, (ra, rb) in halves.items():
                groups.append((f"{c}:A->B", c, cap_subjects(ra, k, f"{c}A"), sorted(rb)))
                groups.append((f"{c}:B->A", c, cap_subjects(rb, k, f"{c}B"), sorted(ra)))
        return groups

    groups = build_groups()
    if args.protocol == "halves":
        pass
    elif args.protocol == "transductive":
        for c, g in tgt_df.groupby("cohort"):
            ids = sorted(g.rec_id)
            groups.append((f"{c}:all", c, ids, ids))
    elif args.protocol == "pooled":
        A = sorted(set().union(*[h[0] for h in halves.values()]))
        B = sorted(set().union(*[h[1] for h in halves.values()]))
        groups.append(("pooled:A->B", "pooled", A, B))
        groups.append(("pooled:B->A", "pooled", B, A))
    rows_by_id = {r.rec_id: r for _, r in tgt_df.iterrows()}

    def target_tensors(ids):
        X = np.concatenate([tgt[r][0] for r in ids])
        y = np.concatenate([tgt[r][2] for r in ids]).astype(np.int64)
        bounds, pos = [], 0
        for r in ids:
            n = len(tgt[r][0])
            bounds.append((pos, pos + n))
            pos += n
        return (torch.from_numpy(X).to(device), torch.from_numpy(y).to(device),
                bounds)

    # ---- baseline records, once
    base_recs = [make_record(rows_by_id[r], tgt[r][2].astype(np.int64),
                             score(head0, tgt[r][0], device))
                 for r in tgt_df.rec_id]
    B = write_metrics(base_recs, os.path.join(out_dir, "baseline"), eval_cfg)
    B["rec_id"] = B["rec_id"].astype(str)
    base_by_id = {r["rec_id"]: r for r in base_recs}

    grid = list(itertools.product(args.tau1, args.tau2, args.lam, args.steps,
                                  args.weighting, args.ema, args.noise,
                                  args.drop, args.cb_floor, args.lr))
    print(f"\n{len(grid)} hyper-parameter combination(s) x {len(args.conditions)} "
          f"condition(s) x {len(groups)} adaptation group(s)\n", flush=True)
    summary, per_cohort, per_rec_frames, run_diags = [], [], [], []
    k_values = args.adapt_k or [0]
    for mode, k in itertools.product(args.conditions, k_values):
        groups = build_groups(k) if args.protocol == "halves" else groups
        k_tag = f"k{k}_" if k else ""
        if mode == "da4":
            hp_list = grid
        else:
            # controls vary only in the step budget and target weight
            hp_list = [g for g in grid if g[:2] == grid[0][:2]
                       and g[4:9] == grid[0][4:9]]
        for (t1, t2, lam, steps, wt, ema, noise, drop, cb, lr) in hp_list:
            hp = HP(tau1=t1, tau2=t2, lam=lam, steps=steps, weighting=wt,
                    ema=ema, noise=noise, drop=drop, cb_floor=cb, lr=lr,
                    batch=args.batch, refresh=args.refresh)
            label = f"{mode}/{args.protocol}/" + k_tag + (
                hp.tag() if mode == "da4" else f"s{steps}_l{lam:g}_lr{lr:g}")
            t0 = time.time()
            recs = {}
            for gname, c, adapt_ids, eval_ids in groups:
                tX, ty, bounds = target_tensors(adapt_ids)
                head, d = adapt_head(head0, src_t, val_t, tX, ty, bounds, hp,
                                     mode, device, args.seed)
                run_diags.append({"condition": label, "group": gname, **d})
                for r in eval_ids:
                    recs[r] = make_record(rows_by_id[r], tgt[r][2].astype(np.int64),
                                          score(head, tgt[r][0], device))
                del tX, ty, head
            ordered = [recs[r] for r in tgt_df.rec_id]
            base_ordered = [base_by_id[r] for r in tgt_df.rec_id]
            safe = label.replace("/", "_")
            A_raw = write_metrics(ordered, os.path.join(out_dir, safe), eval_cfg)
            A_pc = write_metrics(prior_correct(ordered, base_ordered),
                                 os.path.join(out_dir, safe + "_prior"), eval_cfg)
            for A, suffix in ((A_raw, ""), (A_pc, "+prior")):
                A["rec_id"] = A["rec_id"].astype(str)
                row = summarise(A, B, label + suffix)
                dd = pd.DataFrame([x for x in run_diags if x["condition"] == label])
                for k in ("pl_precision_start", "pl_precision_end", "keep_frac",
                          "mean_w", "src_val_acc_base", "src_val_acc"):
                    row[k] = dd[k].mean()
                summary.append(row)
                for c, a in A.groupby("cohort"):
                    b = B[B.cohort == c]
                    r = paired_bootstrap(a, b, "kappa", n_boot=2000, seed=0)
                    per_cohort.append({
                        "condition": label + suffix, "cohort": c, "n": len(a),
                        "kappa_base": b.kappa.mean(), "kappa_delta": r["mean_diff"],
                        "kappa_p": r["p_value"],
                        "acc_delta": a.accuracy.mean() - b.accuracy.mean(),
                        "f1_delta": a.f1_macro.mean() - b.f1_macro.mean(),
                        "f1_N1_delta": a.f1_N1.mean() - b.f1_N1.mean(),
                        "rol_delta": a.rol_ae_min.mean() - b.rol_ae_min.mean()})
                m = A[["rec_id", "cohort", "kappa", "accuracy", "f1_macro",
                       "rol_ae_min"]].merge(
                    B[["rec_id", "kappa", "accuracy", "f1_macro", "rol_ae_min"]],
                    on="rec_id", suffixes=("", "_base"))
                m["condition"] = label + suffix
                per_rec_frames.append(m)
            r0, r1 = summary[-2], summary[-1]
            print(f"  {label:58s} {time.time() - t0:5.0f}s | kappa {r0['kappa_delta']:+.4f}"
                  f" (prior {r1['kappa_delta']:+.4f}, p={r1['kappa_p']:.3f}) | "
                  f"acc {r1['accuracy_delta']*100:+.2f}pp | F1 {r1['f1_macro_delta']:+.4f}"
                  f" | ROL {r1['rol_ae_min_delta']:+.2f} min | w10 {r1['within10_delta']*100:+.1f}pp"
                  f" | keep {r0['keep_frac']:.2f} PLprec {r0['pl_precision_start']:.3f}"
                  f"->{r0['pl_precision_end']:.3f} | srcval {r0['src_val_acc_base']:.4f}"
                  f"->{r0['src_val_acc']:.4f}", flush=True)

    S = pd.DataFrame(summary)
    S.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    PC = pd.DataFrame(per_cohort)
    PC.to_csv(os.path.join(out_dir, "per_cohort.csv"), index=False)
    pd.concat(per_rec_frames, ignore_index=True).to_csv(
        os.path.join(out_dir, "per_recording.csv"), index=False)
    pd.DataFrame(run_diags).to_csv(os.path.join(out_dir, "run_diagnostics.csv"),
                                   index=False)

    print(f"\n{'=' * 100}\nPER-COHORT kappa delta (prior-corrected rows marked +prior)")
    cohorts = sorted(PC.cohort.unique())
    print(f"{'condition':70s}" + "".join(f"{c[:10]:>11s}" for c in cohorts))
    for cond, g in PC.groupby("condition", sort=False):
        cells = "".join(f"{g[g.cohort == c].kappa_delta.iloc[0]:+11.4f}"
                        if (g.cohort == c).any() else f"{'--':>11s}" for c in cohorts)
        print(f"{cond:70s}{cells}")
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
