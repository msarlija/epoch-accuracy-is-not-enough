#!/usr/bin/env bash
# Every result of the paper, in order. Run from the repository root with the
# environment of .env.example. Steps are independent once their inputs exist:
#   bash reproduce.sh            all steps
#   bash reproduce.sh train      one step (data | train | time | adapt | numbers | gssc | figures)
set -euo pipefail
cd "$(dirname "$0")"
PY="uv run --env-file .env python"
STEP="${1:-all}"
run() { [ "$STEP" = all ] || [ "$STEP" = "$1" ]; }
ROOT="$(uv run --env-file .env python -c 'import os; print(os.environ["SLEEP_DATA_ROOT"])')"
PREP="$ROOT/prepared/fs100_band0.2-30"
ANALYSIS="$ROOT/analysis"

# 1. Corpus: raw recordings -> filtered 100 Hz arrays, labels and manifest, then
#    the signal orientation (Section 2.2, Supplementary Section 1): the sign
#    column of the manifest and the orientation check of Supplementary Table S1.
#    Needs the cohort import tables described in data_tables/README.md.
if run data; then
  # ANPHY is released against a central system reference: rebuild Fp1-TP10 and
  # Fp2-TP9 at 250 Hz from the OSF archive first (streams ~85 GB, keeps ~0.5 GB)
  $PY scripts/reextract_anphy.py
  $PY scripts/prepare_data.py --config configs/preprocess.yaml
  $PY scripts/check_prepared.py --config configs/preprocess.yaml
  $PY scripts/set_orientation.py --config configs/preprocess.yaml
  mkdir -p "$ANALYSIS/orientation"
  $PY scripts/orientation_so_spindle.py --data-dir "$PREP" --out-dir "$ANALYSIS/orientation" \
      --apply-sign --split-hpap --per-cohort 30 \
      --datasets CFS YSYW HPAP BOAS MASS MrOS SOF WSC DODO DODH MASS-SS1 ANPHY DREAMS-P DREAMS-S
fi

# 2. Models: the time-aware model (full), the time-naive model (zero) and the
#    branch-removed control (none), 10 folds each; the seven external cohorts
#    scored by every fold model (stage_b/external_v2); and the ten-fold
#    ensembles, written as runs/ens_<arm>_seed0_fold0of10 so that the analysis
#    scripts below score them with --prefix ens --folds 1.
if run train; then
  for ARM in full zero none; do
    for FOLD in 0 1 2 3 4 5 6 7 8 9; do
      $PY scripts/run_experiment.py --config "configs/final_${ARM}.yaml" --fold "$FOLD" \
          --stage-a-epochs 40 --stage-b-epochs 50
    done
  done
  $PY scripts/eval_external.py --arms full zero none --folds 10
  $PY scripts/make_ensemble_run.py --arms full zero none
fi

# 3. Time input (Sections 3.1 to 3.5): Table 1; the fold models against their
#    ensembles (Table 2); the ensemble contrasts (Tables 3, 4, S2, S3, S6, S7,
#    S9, S10); the internal cross-validation block of Table 3 and Table S8
#    (fold models); the confusion matrix of Figure 3; and the two corpus
#    numbers quoted in the text.
if run time; then
  ENS="--prefix ens --folds 1"
  $PY scripts/cohort_table.py                                 # Table 1
  $PY scripts/ensemble_report.py single                       # Table 2
  $PY scripts/ensemble_report.py contrast                     # Tables 3 (external), 4
  $PY scripts/ensemble_report.py contrast --arm-b none        # Table S2
  $PY scripts/ensemble_report.py size                         # Table S6
  $PY scripts/ensemble_report.py onsets                       # Section 3.4, Table S10
  $PY scripts/ablation_report.py --folds 10 --arms full zero  # Table 3, internal block
  $PY scripts/latency_agreement_report.py $ENS --splits external_v2    # Tables S3, S7
  $PY scripts/latency_agreement_report.py --folds 10 --splits test     # Table S8
  $PY scripts/rem_persistence_check.py $ENS --split external_v2        # Table S9
  $PY scripts/confusion_matrix.py $ENS --splits external_v2            # Figure 3 values
  $PY scripts/corpus_numbers.py
fi

# 4. Label-free adaptation (DA-G, DA-MC and the single-anchor variant of
#    Supplementary Table S13) and the supervised reference (DA-O), on the ten
#    time-aware fold models, with the per-epoch probabilities dumped so that
#    the adapted fold models can be ensembled.
if run adapt; then
  PROT="loso loso1 loso2 loso5 loso10 loso15 cohort"
  $PY scripts/domain_adapt.py --folds 10 --strategy da1 da2 --protocol $PROT \
      --n2-source pred-conf --n2-conf 0.7 --where pre --transform affine \
      --prior-correct --min-epochs 200 --out-name da12 --dump-proba
  $PY scripts/domain_adapt.py --folds 10 --strategy da2 --protocol $PROT \
      --n2-source pred-conf --n2-conf 0.7 --where pre --transform multicluster \
      --clusters n2 wake rem --mc-missing drop --prior-correct --min-epochs 50 --out-name da3 --dump-proba
  $PY scripts/domain_adapt.py --folds 10 --strategy da2 --protocol loso loso1 loso2 loso5 loso10 loso15 \
      --n2-source true --n2-conf 0.7 --where pre --transform multicluster \
      --clusters n2 wake rem --mc-missing drop --prior-correct --min-epochs 50 --out-name da3_oracle --dump-proba
  for FOLD in 0 1 2 3 4 5 6 7 8 9; do
    $PY scripts/da4_self_train.py --fold "$FOLD" --conditions oracle --adapt-k 1 2 5 10 15 0 \
        --steps 2000 --lr 1e-4 --tag da_o --dump-proba
  done
fi

# 5. Every adaptation number of the paper: the ensembles (Tables 6, 7, S5, S13,
#    S14, Figure 4 and Section 3.8) and the fold-model values behind the
#    class-prior comparison of Supplementary Table S4.
if run numbers; then
  $PY scripts/paper_da_numbers.py
  $PY scripts/ensemble_da_numbers.py
fi

# 6. The open-source GSSC stager on the same external recordings (Table 5,
#    Supplementary Tables S11 and S12) and its parameter count. Needs `uv sync --extra gssc`.
if run gssc; then
  $PY scripts/gssc_param_count.py
  $PY scripts/gssc_score.py
  $PY scripts/gssc_paper_table.py --prefix ens --folds 1     # ensembles
  $PY scripts/gssc_paper_table.py                            # fold models (column Single of Table 5)
  $PY scripts/gssc_confusion.py                              # confusion matrices (Table S12)
fi

# 7. Figures 2 to 4 (Figure 1 is drawn in the manuscript source).
if run figures; then
  $PY scripts/find_early_rem_examples.py --prefix ens --folds 1 --plot --paper --wide \
      --acc-delta 0.05 --tolerance 30 \
      --only DODH_1da3544e-dc5c-5795-adc3-f5068959211f --outdir figures/paper/wide
  $PY scripts/make_paper_figures.py confusion --wide --prefix ens --folds 1
  $PY scripts/make_paper_figures.py kcurve --wide
fi
