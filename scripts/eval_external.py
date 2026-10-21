# -*- coding: utf-8 -*-
"""Re-score the external cohorts with checkpoints that are already trained.

Adding an external cohort after training does NOT require retraining: external
data never enters train/val/test, so the model is unchanged and only inference
is missing. This loads each finished run's stage_b.pt, evaluates the CURRENT
external cohort list from the config, and writes the metrics beside the
originals under a new directory name.

Uses the same evaluate_split / write_metrics path as run_experiment.py, so the
outputs are byte-comparable with the tables produced during training. External
recordings are scored UNTRIMMED, exactly as in training.

``head_time_skip`` is taken from the CHECKPOINT rather than the config: runs
predating E7 were written before that key existed, and rebuilding them from
their own resolved_config.yaml silently produces the wrong head width.

    python scripts/eval_external.py --prefix final --arms full zero none \
        --folds 10 --out-name external_v2
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config                       # noqa: E402
from data import manifest as mf                      # noqa: E402
from data.sampling import apply_cohort_caps           # noqa: E402
from models.model import (build_model,                # noqa: E402
                          reconcile_head_time_skip)
from scripts.run_experiment import (evaluate_split,   # noqa: E402
                                    write_metrics)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", default=os.path.expanduser(os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging") + "/runs"))
    ap.add_argument("--prefix", default="final")
    ap.add_argument("--arms", nargs="+", default=["full", "zero", "none"])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--config-prefix", default="final",
                    help="config family supplying the external cohort list and "
                         "normalisation, as configs/<config-prefix>_<arm>.yaml. "
                         "Separate from --prefix because the run directories "
                         "keep the name they were trained under (final_*) "
                         "while the configs were renamed (final_*).")
    ap.add_argument("--config", default="", help="explicit config path, "
                    "overriding --config-prefix")
    ap.add_argument("--out-name", default="external_v2",
                    help="subdirectory under <run>/stage_b/ to write into")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--stage", default="stage_b", choices=("stage_a", "stage_b"),
                    help="which checkpoint to score. stage_a is the epoch-level "
                         "model (with_sequence=False), evaluated in its own "
                         "right and not merely as an initialisation.")
    ap.add_argument("--force", action="store_true",
                    help="re-score folds that already have output")
    args = ap.parse_args()

    for arm in args.arms:
        cfg_path = args.config or f"configs/{args.config_prefix}_{arm}.yaml"
        cfg = load_config(cfg_path)
        ext = cfg.get("external_cohorts") or []
        data_dir = cfg["data_dir"]
        manifest = mf.load_manifest(data_dir)
        if cfg.get("qc", {}).get("filter", True):
            manifest = manifest[manifest["qc_pass"] == 1].reset_index(drop=True)
        split_df = manifest[manifest["cohort"].isin(ext)].reset_index(drop=True)
        # Apply cohort_caps to the external set too. run_experiment.py does NOT
        # (external_df is taken before apply_cohort_caps), which is why WSC
        # appeared as 102 and DREAMS as 46 despite a cap of 40. Capping here is
        # subject-level and seeded, so it is reproducible and never splits a
        # subject's nights across the boundary.
        caps = cfg.get("cohort_caps") or {}
        if caps:
            before = len(split_df)
            split_df = apply_cohort_caps(split_df, caps, seed=cfg.get("seed", 0))
            split_df = split_df.reset_index(drop=True)
            if len(split_df) != before:
                print(f"  cohort caps: {before} -> {len(split_df)} recordings",
                      flush=True)
        missing = sorted(set(ext) - set(manifest["cohort"]))
        if missing:
            raise SystemExit(f"external cohorts absent from the manifest: {missing}")

        norm = cfg.get("normalization", {})
        eval_cfg = cfg.get("eval", {})
        print(f"\n=== {args.prefix}_{arm}: {len(split_df)} external recordings, "
              f"{split_df.cohort.nunique()} cohorts ===", flush=True)

        for f in range(args.folds):
            run_dir = f"{args.runs_root}/{args.prefix}_{arm}_seed0_fold{f}of10"
            ckpt_path = os.path.join(run_dir, f"{args.stage}.pt")
            if not os.path.exists(ckpt_path):
                print(f"  fold {f}: no stage_b.pt, skipping", flush=True)
                continue
            done_marker = os.path.join(run_dir, args.stage, args.out_name,
                                       "per_recording_metrics.csv")
            if os.path.exists(done_marker) and not args.force:
                print(f"  fold {f}: already done, skipping", flush=True)
                continue
            ckpt = torch.load(ckpt_path, map_location="cpu")
            run_cfg = load_config(os.path.join(run_dir, "resolved_config.yaml"))
            mcfg = reconcile_head_time_skip(dict(run_cfg["model"]),
                                        ckpt["state_dict"],
                                        os.path.basename(run_dir))
            mcfg["with_sequence"] = args.stage == "stage_b"
            model = build_model(mcfg).to(args.device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            with torch.no_grad():
                records = evaluate_split(
                    model, split_df, data_dir, norm.get("mode", "fixed"), norm,
                    cfg.get("c_max", 2), args.device,
                    pred_dir=os.path.join(run_dir, args.stage,
                                          f"predictions_{args.out_name}"),
                    min_label_purity=(cfg.get("purity") or {}).get("min_label_purity"),
                    chunk_epochs=eval_cfg.get("inference_chunk_epochs", 512),
                    trim_wake=None,          # external is scored untrimmed
                    seed=cfg.get("seed", 0),
                    tso_norm=cfg.get("tso_norm", "elapsed"))
            joined = write_metrics(records,
                                   os.path.join(run_dir, args.stage, args.out_name),
                                   eval_cfg)
            print(f"  fold {f}: {len(joined)} recordings -> "
                  f"kappa {joined.kappa.mean():.4f}  acc {joined.accuracy.mean():.4f}",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
