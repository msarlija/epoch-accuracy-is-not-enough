# -*- coding: utf-8 -*-
"""The open-source GSSC stager (Hanna and Floel 2023, package gssc 0.0.9, networks
sig_net_v1 and gru_net_v1) on the external recordings, for Table 5 and
Supplementary Table S11 (scripts/gssc_paper_table.py).

GSSC receives the prepared signal the models of this study read: every channel
that passed quality control, at 100 Hz, without EOG, and combines the channels
with its own consensus rule. Its hypnogram is evaluated with the paper's
definitions. Recordings already in the output table are skipped, so the run can
be resumed. WSC is scored too; the table script excludes it because GSSC was
trained on it.

    uv sync --extra gssc
    python scripts/gssc_score.py [--cohorts ...]
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config                           # noqa: E402
from data import manifest as mf                          # noqa: E402
from data.sampling import apply_cohort_caps              # noqa: E402
from eval.epoch_metrics import epoch_metrics             # noqa: E402
from eval.macrostructure import record_macrostructure    # noqa: E402

ROOT = os.path.expanduser(os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging"))
OUT = f"{ROOT}/analysis/gssc_external_perrec.csv"


def good_channels(row):
    f = row["qc_channel_flags"]
    f = json.loads(f) if isinstance(f, str) else f
    return [i for i, v in enumerate(f) if v == 1] if f is not None and len(f) else [0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/final_full.yaml")
    ap.add_argument("--cohorts", nargs="*", default=None, help="default: the config's external cohorts")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    import mne
    import torch
    from importlib.resources import files
    from gssc.infer import EEGInfer
    warnings.filterwarnings("ignore")
    mne.set_log_level("ERROR")
    # the pretrained networks ship as pickled modules inside the gssc package
    net = torch.load(files("gssc.nets").joinpath("sig_net_v1.pt"), weights_only=False, map_location="cpu")
    con = torch.load(files("gssc.nets").joinpath("gru_net_v1.pt"), weights_only=False, map_location="cpu")
    inf = EEGInfer(net=net, con_net=con, use_cuda=False)

    def stage(x, chans):
        names = [f"EEG{i}" for i in chans]
        raw = mne.io.RawArray(x[:, chans].T.astype(np.float64) * 1e-6, mne.create_info(names, 100.0, "eeg"))
        hyp, _ = inf.mne_infer(raw, eeg=names, eog=[], filter=True)
        return np.asarray(hyp, int)

    cfg = load_config(args.config)
    data_dir = cfg["data_dir"]
    cohorts = args.cohorts or cfg["external_cohorts"]
    man = mf.load_manifest(data_dir)
    man = man[(man.qc_pass == 1) & man.cohort.isin(cohorts)]
    ext = apply_cohort_caps(man, cfg.get("cohort_caps") or {}, seed=cfg.get("seed", 0)).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set(pd.read_csv(args.out).rec_id) if os.path.exists(args.out) else set()
    t0 = time.time()
    for i, r in ext.iterrows():
        if r.rec_id in done:
            continue
        a = mf.open_arrays(data_dir, r.rec_id, mmap=True)
        y = np.asarray(a["y"], int)
        p = stage(np.asarray(a["x"]), good_channels(r))
        n = min(len(y), len(p))                        # GSSC leaves the last partial epoch unscored
        yp = np.full(len(y), -1)
        yp[:n] = np.where(y[:n] < 0, -1, p[:n])
        k = (y >= 0) & (np.arange(len(y)) < n)
        m = epoch_metrics(y[k], yp[k])
        mac = record_macrostructure(y, yp)
        row = dict(rec_id=r.rec_id, cohort=r.cohort, subject_id=str(r.subject_id), kappa=m["kappa"],
                   accuracy=m["accuracy"], f1_macro=m["f1_macro"], rol_true_min=mac["rol_true_min"],
                   rol_pred_min=mac["rol_pred_min"], rol_ae_min=mac["rol_ae_min"], n_chan=len(good_channels(r)))
        pd.DataFrame([row]).to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
        if i % 20 == 0:
            print(f"{i + 1}/{len(ext)} {time.time() - t0:.0f}s", flush=True)
    print(f"GSSC scored {len(ext)} recordings -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
