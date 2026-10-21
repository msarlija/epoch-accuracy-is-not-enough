# Epoch accuracy is not enough

Code for "Epoch accuracy is not enough: time-aware sleep staging from frontal
EEG across fourteen cohorts, with label-free adaptation to new sites".

The repository contains what is needed to reproduce the reported results:
preprocessing, the model and its training, external scoring, the analyses
behind every table and figure, and the adaptation study. `reproduce.sh` runs
them in order.

## Setup

Python 3.12 with [uv](https://docs.astral.sh/uv/):

```
uv sync
cp .env.example .env    # set SLEEP_RAW_ROOT and SLEEP_DATA_ROOT
```

`SLEEP_RAW_ROOT` is the parent directory of the raw cohorts, and
`SLEEP_DATA_ROOT` receives the prepared corpus, the trained models and the
analysis outputs. Training uses one GPU.

## Data

The fourteen cohorts come from their public or access-controlled sources
(see the Data availability statement of the paper). The subject-level import
tables that name the recordings are not redistributed; `data_tables/README.md`
lists where each comes from and which columns the readers expect.

## Reproduction

```
bash reproduce.sh            # everything
bash reproduce.sh data       # corpus preparation
bash reproduce.sh train      # 3 models x 10 folds, then external scoring
bash reproduce.sh time       # time-input results (Tables 1-4, S1-S7)
bash reproduce.sh adapt      # adaptation runs (DA-1 to DA-3, DA-O)
bash reproduce.sh numbers    # adaptation results (Tables 6, 7, S8-S10)
bash reproduce.sh yasa       # open-source YASA comparison (Table 5, S11)
bash reproduce.sh figures    # Figures 2-6
```

The time-aware model is `configs/final_full.yaml`. The time-naive control,
whose time input is held at zero, is `configs/final_zero.yaml`. The second
control without the time branch is `configs/final_none.yaml`.

## Layout

```
configs/      model, training and preprocessing configuration
data/         cohort readers, preprocessing, quality control, datasets
models/       epoch encoder, channel attention, time branch, sequence model
training/     two-stage training
eval/         epoch metrics, macrostructure endpoints, paired bootstrap
scripts/      entry points and the analyses behind the paper
```
