# -*- coding: utf-8 -*-
"""Regenerate the MrOS / SOF import tables from what is actually on disk.

The tables shipped with the repo name a *different* set of recordings than the
S3 sync actually pulled down: of the 50 MrOS ids listed, 8 had an EDF and none
had an annotation; of the 49 SOF ids, 4 matched a real EDF+XML pair. Because
``data/readers/nsrr_xml.py`` drives strictly off the table (and ``limit`` just
takes the first N rows), the table has to describe the recordings that exist or
the cohort yields almost nothing.

This script derives each table from three facts on disk — which EDFs are here,
which annotation XMLs are here, and the NSRR metadata release — and writes the
intersection with the age/sex columns ``configs/preprocess.yaml`` names. It is
a script rather than a shell one-liner because "which recordings constitute the
corpus" is a provenance question that has to stay re-derivable.

Usage:
    python scripts/build_nsrr_import_tables.py --cohort mros
    python scripts/build_nsrr_import_tables.py --cohort sof
    python scripts/build_nsrr_import_tables.py --cohort all --check
"""

import argparse
import glob
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

log = logging.getLogger("build_nsrr_import_tables")


def _ids_from(pattern: str, prefix: str, suffix: str) -> set:
    """Stems of the files matching ``pattern``, with prefix/suffix stripped."""
    out = set()
    for path in glob.glob(pattern):
        base = os.path.basename(path)
        if base.startswith(prefix) and base.endswith(suffix):
            out.add(base[len(prefix):len(base) - len(suffix)])
    return out


def build_mros(raw_root: str) -> pd.DataFrame:
    """MrOS visit 1: EDFs in temp/ (lowercase ids), annotations in the full
    NSRR export tree, metadata from the harmonized dataset release."""
    edf_ids = _ids_from(f"{raw_root}/MrOS/temp/mros-visit1-*.edf",
                        "mros-visit1-", ".edf")
    ann_ids = _ids_from(
        f"{raw_root}/MrOS/mros/polysomnography/annotations-events-nsrr/visit1/"
        f"mros-visit1-*-nsrr.xml", "mros-visit1-", "-nsrr.xml")
    ids = sorted(edf_ids & ann_ids)
    log.info("MrOS: %d EDFs, %d annotation XMLs, %d usable pairs",
             len(edf_ids), len(ann_ids), len(ids))
    if not ids:
        raise SystemExit("MrOS: no EDF/XML pairs found — check SLEEP_RAW_ROOT")

    meta_path = f"{raw_root}/MrOS/mros/datasets/mros-visit1-harmonized-0.6.0.csv"
    meta = pd.read_csv(meta_path, usecols=["nsrrid", "nsrr_age", "nsrr_sex"])
    # The metadata release uses uppercase ids; every file on disk is lowercase.
    # /mnt/c is case-insensitive so uppercase happens to work today, but the
    # table has to survive the corpus moving to a case-sensitive filesystem.
    meta["nsrrid"] = meta["nsrrid"].astype(str).str.lower()

    df = pd.DataFrame({"nsrrid": ids}).merge(meta, on="nsrrid", how="left")
    missing = int(df["nsrr_age"].isna().sum())
    if missing:
        raise SystemExit(
            f"MrOS: {missing} of {len(df)} recordings have no age in "
            f"{meta_path} — wrong metadata version?")
    return df


def build_sof(raw_root: str) -> pd.DataFrame:
    """SOF visit 8: EDFs and annotations both in temp/, metadata from the
    visit-8 dataset release. Only ~50 of the EDFs have an annotation."""
    edf_ids = _ids_from(f"{raw_root}/SOF/temp/sof-visit-8-*.edf",
                        "sof-visit-8-", ".edf")
    ann_ids = _ids_from(f"{raw_root}/SOF/temp/sof-visit-8-*-nsrr.xml",
                        "sof-visit-8-", "-nsrr.xml")
    ids = sorted(edf_ids & ann_ids, key=int)
    log.info("SOF: %d EDFs, %d annotation XMLs, %d usable pairs",
             len(edf_ids), len(ann_ids), len(ids))
    if not ids:
        raise SystemExit("SOF: no EDF/XML pairs found — check SLEEP_RAW_ROOT")

    meta_path = f"{raw_root}/SOF/sof/datasets/sof-visit-8-dataset-0.7.0.csv"
    meta = pd.read_csv(meta_path, usecols=["sofid", "v8age"], low_memory=False)
    meta["sofid"] = meta["sofid"].astype(int)

    # Filenames are zero-padded to 5 digits; the table stores the bare integer
    # and configs/preprocess.yaml re-pads it via id_format: "05d".
    df = pd.DataFrame({"sofid": [int(i) for i in ids]}).merge(
        meta, on="sofid", how="left")
    missing = int(df["v8age"].isna().sum())
    if missing:
        raise SystemExit(
            f"SOF: {missing} of {len(df)} recordings have no age in "
            f"{meta_path} — wrong metadata version?")
    # SOF (Study of Osteoporotic Fractures) enrolled women only; the release
    # therefore carries no sex column. Stating it keeps the manifest's `sex`
    # field populated for these recordings instead of 50 nulls.
    df["sex"] = "F"
    return df


BUILDERS = {"mros": (build_mros, "mros_import.csv"),
            "sof": (build_sof, "sof_import.csv")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True,
                        choices=sorted(BUILDERS) + ["all"])
    parser.add_argument("--check", action="store_true",
                        help="report what would be written without writing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    raw_root = os.environ.get("SLEEP_RAW_ROOT")
    if not raw_root:
        raise SystemExit("SLEEP_RAW_ROOT is not set; see .env.example")

    cohorts = sorted(BUILDERS) if args.cohort == "all" else [args.cohort]
    for cohort in cohorts:
        builder, filename = BUILDERS[cohort]
        df = builder(raw_root)
        out = os.path.join(REPO, "data_tables", filename)
        if args.check:
            log.info("%s: would write %d rows to %s", cohort, len(df), out)
        else:
            df.to_csv(out, index=False)
            log.info("%s: wrote %d rows to %s", cohort, len(df), out)
        log.info("%s: columns %s; first ids %s", cohort, list(df.columns),
                 df.iloc[:3, 0].tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
