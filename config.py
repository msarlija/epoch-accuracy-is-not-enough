# -*- coding: utf-8 -*-
"""Plain-YAML config loading (deliberately not a framework).

A config file may name parent files in ``defaults`` (string or list, paths
relative to the file); parents are deep-merged in order, the child wins.
``load_config`` returns the merged dict; ``save_resolved`` writes it next
to a run's outputs (§3.16 — resolved config saved with each run).

Paths are kept out of the configs themselves via ``${VAR}`` expansion, so the
same YAML works on any machine (the corpus lives on a Windows drive under WSL
here, somewhere else on the next box). Two variables come from the environment
— see ``.env.example`` — and ``${SLEEP_REPO}`` is supplied automatically.
An unset variable is an error, not a literal path: silently writing a corpus
into a directory named ``${SLEEP_DATA_ROOT}`` is the failure mode this
exists to prevent.
"""

import copy
import os
import re
from typing import Any, Dict, List, Union

import yaml

_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: Supplied without any setup: this file sits at the repo root, which is also
#: how every entrypoint derives it for ``sys.path``. Lets the configs reference
#: in-repo data tables without depending on the process CWD.
_IMPLICIT = {"SLEEP_REPO": os.path.dirname(os.path.abspath(__file__))}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if (key in out and isinstance(out[key], dict)
                and isinstance(val, dict)):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _expand(node: Any) -> Any:
    """Recursively expand ``${VAR}`` and ``~`` in every string leaf."""
    if isinstance(node, str):
        def _sub(match: "re.Match[str]") -> str:
            name = match.group(1)
            value = os.environ.get(name, _IMPLICIT.get(name))
            if value is None:
                raise KeyError(
                    f"config references ${{{name}}} but it is not set in the "
                    f"environment; see .env.example (and pass "
                    f"`uv run --env-file .env`)")
            return value
        return os.path.expanduser(_VAR_RE.sub(_sub, node))
    if isinstance(node, dict):
        return {k: _expand(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand(v) for v in node]
    return node


def load_config(path: str) -> Dict[str, Any]:
    """Load ``path``, merge its ``defaults`` ancestry, then expand ``${VAR}``.

    Expansion happens once on the fully merged result, so a parent may
    reference a variable that only matters for values the child overrides.
    """
    return _expand(_load_raw(path))


def _load_raw(path: str) -> Dict[str, Any]:
    path = os.path.abspath(path)
    # utf-8-sig: configs contain non-ASCII (§ references) and may carry a BOM;
    # Windows' default locale codec must not be used
    with open(path, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f) or {}

    defaults: Union[str, List[str], None] = cfg.pop("defaults", None)
    if defaults is None:
        return cfg
    if isinstance(defaults, str):
        defaults = [defaults]
    merged: Dict[str, Any] = {}
    for parent in defaults:
        parent_path = os.path.join(os.path.dirname(path), parent)
        merged = _deep_merge(merged, _load_raw(parent_path))
    return _deep_merge(merged, cfg)


def save_resolved(cfg: Dict[str, Any], run_dir: str,
                  name: str = "resolved_config.yaml") -> str:
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False,
                       allow_unicode=True)
    return path


def external_cohorts(config_path: str = None) -> list:
    """The external (evaluation-only) cohort list of the shipped final config.

    Analysis scripts read evaluation tables that were written for whatever
    ``external_cohorts`` said at scoring time; a cohort removed from the config
    later (STAGES-MSMI, 2026-10) would otherwise linger in every pooled number.
    Filtering on this list makes the config the single place a cohort is added
    or dropped from the external set. Parsed from the raw YAML so no
    environment variables are needed.
    """
    import yaml
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "configs", "final_base.yaml")
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cohorts = raw.get("external_cohorts")
    if not cohorts:
        raise KeyError(f"{config_path} has no external_cohorts")
    return list(cohorts)
