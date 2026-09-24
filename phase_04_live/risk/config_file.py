"""
phase_04_live/risk/config_file.py

File-backed live update channel for LiveRiskConfig.

LiveRiskConfig's own docstring says it's "meant to be adjusted at
runtime by the dashboard... without restarting the live process."
This module is what makes that real: the dashboard (a separate
process) writes sizing settings to a small JSON file, and the running
loop re-reads it on every candle cycle, mutating its LiveRiskConfig
in place if the file's contents changed.

This does NOT replace LiveRiskConfig.validate() -- a bad value
written here is caught by validate() the next time
compute_position_size() runs, same as any other mutation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from phase_04_live.risk.sizing import LiveRiskConfig

_MUTABLE_FIELDS = (
    "sizing_mode",
    "fixed_lot_size",
    "risk_percent_per_trade",
    "max_risk_percent_per_trade",
    "reward_multiple_override",
    "max_lots_per_trade",
)


def apply_config_file(
    config: LiveRiskConfig,
    path: Path,
) -> Optional[str]:
    """
    If `path` exists and contains valid JSON, apply any of
    _MUTABLE_FIELDS present in it onto `config`, in place.

    Returns None on success (including "file doesn't exist yet" --
    that's not an error, just "no update pending"), or a short error
    string if the file exists but is malformed/invalid. The caller
    decides whether to log and continue (never crash the trading loop
    over a malformed config file).
    """
    if not path.exists():
        return None

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return f"could not read/parse {path}: {exc}"

    if not isinstance(raw, dict):
        return f"{path} did not contain a JSON object"

    for field in _MUTABLE_FIELDS:
        if field in raw:
            setattr(config, field, raw[field])

    try:
        config.validate()
    except ValueError as exc:
        return f"{path} contained an invalid value: {exc}"

    return None


__all__ = ["apply_config_file"]