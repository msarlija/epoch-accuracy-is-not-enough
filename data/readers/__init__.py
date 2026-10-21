# -*- coding: utf-8 -*-
"""Cohort reader registry.

Each reader module exposes ``iter_recordings(spec, cfg)`` yielding
:class:`~data.readers.base.RawRecording`. Readers are imported lazily so a
missing optional dependency (wfdb, h5py, mne) only breaks the cohorts that
need it.
"""

import importlib

from .base import (Hypnogram, RawRecording, STAGE_NAMES, UNKNOWN, VALID_STAGES,
                   WAKE, N1, N2, N3, REM)

_READER_MODULES = {
    "mass": "data.readers.mass",
    "nsrr_xml": "data.readers.nsrr_xml",
    "dreams": "data.readers.dreams",
    "anphy": "data.readers.anphy",
    "boas": "data.readers.boas",
    "ysyw": "data.readers.ysyw",
    "wsc": "data.readers.wsc",
    "dodh": "data.readers.dodh",
}


def get_reader(name: str):
    """Return the ``iter_recordings`` callable for a reader name."""
    try:
        module = importlib.import_module(_READER_MODULES[name])
    except KeyError:
        raise KeyError(f"Unknown reader '{name}'. "
                       f"Known: {sorted(_READER_MODULES)}") from None
    return module.iter_recordings
