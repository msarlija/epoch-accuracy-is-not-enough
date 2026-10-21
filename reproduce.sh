#!/usr/bin/env bash
# Every result of the paper, in order. Run from the repository root with the
# environment of .env.example. Steps are independent once their inputs exist:
#   bash reproduce.sh            all steps
#   bash reproduce.sh train      one step (data | train | time | adapt | numbers | yasa | figures)
set -euo pipefail
cd "$(dirname "$0")"
PY="uv run --env-file .env python"
STEP="${1:-all}"
run() { [ "$STEP" = all ] || [ "$STEP" = "$1" ]; }

# 1. Corpus: raw recordings -> filtered 100 Hz arrays, labels and manifest.
#    Needs the cohort import tables described in data_tables/README.md.
if run data; then
  $PY scripts/prepare_data.py --config configs/preprocess.yaml
  $PY scripts/check_prepared.py --config configs/preprocess.yaml
fi

# 2. Models: the time-aware model (full), the time-naive model (zero) and the
#    branch-removed control (none), 10 folds each, then the seven external
#    cohorts scored by every fold model (written to stage_b/external_v2).
if run train; then
  for ARM in full zero none; do
    for FOLD in 0 1 2 3 4 5 6 7 8 9; do
      $PY scripts/run_experiment.py --config "configs/final_${ARM}.yaml" --fold "$FOLD" \
          --stage-a-epochs 40 --stage-b-epochs 50
    done
  done
  $PY scripts/eval_external.py --arms full zero none --folds 10
fi

# 3. Time input: Table 1, epoch-level agreement, REM-onset latency, false early
#    REM, the persistence rule and the confusion matrices (main text and
#    Supplementary Tables S1-S7; the branch-removed control is --arm-b none).
if run time; then
  $PY scripts/cohort_table.py
  $PY scripts/ablation_report.py --folds 10
  $PY scripts/latency_agreement_report.py
  $PY scripts/latency_agreement_report.py --arm-b none
  $PY scripts/failure_rate_analysis.py --folds 10
  $PY scripts/rem_persistence_check.py
  $PY scripts/rem_persistence_check.py --arm-b none
  $PY scripts/confusion_matrix.py
fi

# 4. Label-free adaptation (DA-1 to DA-3) and the supervised reference (DA-O),
#    on the ten time-aware fold models.
if run adapt; then
  PROT="loso loso1 loso2 loso5 loso10 loso15 cohort recording"
  $PY scripts/domain_adapt.py --folds 10 --strategy da1 da2 --protocol $PROT \
      --n2-source pred-conf --n2-conf 0.7 --where pre --transform affine \
      --prior-correct --min-epochs 200 --out-name da12
  $PY scripts/domain_adapt.py --folds 10 --strategy da2 --protocol $PROT \
      --n2-source pred-conf --n2-conf 0.7 --where pre --transform multicluster \
      --clusters n2 wake rem --mc-missing drop --prior-correct --min-epochs 50 --out-name da3
  $PY scripts/domain_adapt.py --folds 10 --strategy da2 --protocol loso loso1 loso2 loso5 loso10 loso15 \
      --n2-source true --n2-conf 0.7 --where pre --transform multicluster \
      --clusters n2 wake rem --mc-missing drop --prior-correct --min-epochs 50 --out-name da3_oracle
  for FOLD in 0 1 2 3 4 5 6 7 8 9; do
    $PY scripts/da4_self_train.py --fold "$FOLD" --conditions oracle --adapt-k 1 2 5 10 15 0 \
        --steps 2000 --lr 1e-4 --tag da_o
  done
fi

# 5. Every adaptation number of the paper (Tables 6, 7, S8-S10 and the text).
if run numbers; then
  $PY scripts/paper_da_numbers.py
fi

# 6. The open-source YASA stager on the same external recordings (Table 5,
#    Supplementary Table S11). Needs `uv sync --extra yasa` and the OpenMP runtime.
if run yasa; then
  $PY scripts/yasa_comparison.py
fi

# 7. Figures 2 to 6 (Figure 1 is drawn in the manuscript source).
if run figures; then
  $PY scripts/fig_stage_vs_time.py --wide
  $PY scripts/make_paper_figures.py forest --wide
  $PY scripts/find_early_rem_examples.py --plot --paper --wide \
      --only DODO_79e8a04e-0fdd-53ae-bedc-d49d08e29103 --outdir figures/paper/wide
  $PY scripts/make_paper_figures.py confusion --wide
  $PY scripts/make_paper_figures.py kcurve --wide
fi
