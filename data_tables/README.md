# Cohort import tables

`configs/preprocess.yaml` points each cohort reader at a CSV in this directory
that names the recordings to ingest and carries the per-subject age and sex used
for stratification and for the age-matched analyses.

**The CSVs themselves are deliberately not distributed with this repository.**
They contain subject-level data — identifiers, age, sex, and for some cohorts
clinical variables (AHI, BMI, blood pressure, lipid panels, diagnoses) — drawn
from sources held under data use agreements that do not permit redistribution.
Obtaining them is a matter of applying to each source in the normal way, not of
finding a copy.

| table | cohort | source | access |
|---|---|---|---|
| `wsc-dataset-0.3.0.csv` | WSC | [sleepdata.org/datasets/wsc](https://sleepdata.org/datasets/wsc) | NSRR DUA |
| `cfs_import.csv` | CFS | [sleepdata.org/datasets/cfs](https://sleepdata.org/datasets/cfs) | NSRR DUA |
| `mros_import.csv` | MrOS | [sleepdata.org/datasets/mros](https://sleepdata.org/datasets/mros) | NSRR DUA |
| `sof_import.csv` | SOF | [sleepdata.org/datasets/sof](https://sleepdata.org/datasets/sof) | NSRR DUA |
| `homepap_import.csv` | HomePAP | [sleepdata.org/datasets/homepap](https://sleepdata.org/datasets/homepap) | NSRR DUA |
| `mass_import.csv`, `mass_ss1_import.csv` | MASS | [ceams-carsm.ca/mass](http://ceams-carsm.ca/mass/) | application |
| `boas_import.csv` | BOAS | [doi.org/10.34808/9d5s-jm43](https://doi.org/10.34808/9d5s-jm43) | open |
| `YSYW-age-sex.csv` | YSYW (PhysioNet Challenge 2018) | [physionet.org/content/challenge-2018](https://physionet.org/content/challenge-2018/) | open |

## Expected columns

A reader needs only the columns `configs/preprocess.yaml` names for its cohort;
extra columns are ignored. The minimum is a recording identifier plus `Age` and
`Sex`. See the `import_csv` / `participants_csv` / `age_sex_csv` keys there for
the exact column names per cohort, and `data/readers/` for how each is parsed.

`scripts/build_nsrr_import_tables.py` regenerates the MrOS and SOF tables from
an NSRR metadata release plus whatever EDF/XML pairs are actually on disk, which
is the reliable way to keep a table describing recordings that exist rather than
recordings that were requested.

## Why this matters for reproduction

Every cohort in this study is access-controlled except BOAS, YSYW, DREAMS,
DOD-H and DOD-O. Exact reproduction therefore requires the same approvals we
held. The code path is fully specified here; the data is not ours to pass on.
