"""
strategy/zones.py

Detects Supply and Demand zones from confirmed structural breaks
(INITIAL_BREAK / BOS / CHOCH) produced by strategy/structure.py.

Zone rules:
    Origin:
        The full cluster of consecutive candles opposite in color to
        the impulsive move that produced the structural break —
        walked backward from the break candle, past the entire
        impulsive run, then collecting the whole opposite-colored
        base immediately before it.

    Boundaries:
        price_top    = highest High in the origin cluster
        price_bottom = lowest Low in the origin cluster

    Mitigation:
        A zone is mitigated the instant any later candle's wick
        (High/Low) overlaps [price_bottom, price_top] — a touch,
        not a close.

Candle color:
    bullish : close > open
    bearish : close < open
    neutral : close == open (doji)

A neutral candle sitting where the base cluster should start means
no clean opposite-colored base exists there. Rather than guess which
side it belongs to, no zone is produced for that break. This is a
conservative default and worth revisiting once real data shows how
often it happens.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import pandas as pd

from strategy.structure import StructureBreak, Trend


class ZoneType(Enum):
    DEMAND = "DEMAND"
    SUPPLY = "SUPPLY"


@dataclass
class Zone:
    zone_type: ZoneType
    price_top: float
    price_bottom: float
    origin_start: pd.Timestamp
    origin_end: pd.Timestamp
    formed_from_break_at: pd.Timestamp
    mitigated: bool = False
    mitigated_at: Optional[pd.Timestamp] = None

    def __repr__(self):
        return (
            f"Zone({self.zone_type.value}, "
            f"top={self.price_top}, bottom={self.price_bottom}, "
            f"origin={self.origin_start.strftime('%m-%d %H:%M')}"
            f"->{self.origin_end.strftime('%m-%d %H:%M')}, "
            f"mitigated={self.mitigated})"
        )


def _candle_color(row) -> str:
    if row["close"] > row["open"]:
        return "bullish"
    if row["close"] < row["open"]:
        return "bearish"
    return "neutral"


def _find_origin_cluster(df: pd.DataFrame, break_idx: int, impulse_color: str):
    """
    Walk backward from the break candle to find the origin cluster:
    the full run of consecutive candles opposite to impulse_color
    sitting immediately before the impulsive run that led to the break.

    Returns (start_idx, end_idx) inclusive positions into df, or None
    if no valid cluster is found (impulse consumes the start of the
    data, or a neutral/same-color candle blocks the base).
    """
    base_color = "bearish" if impulse_color == "bullish" else "bullish"

    i = break_idx

    # Walk back through the impulsive run itself.
    while i >= 0 and _candle_color(df.iloc[i]) == impulse_color:
        i -= 1

    if i < 0:
        return None

    if _candle_color(df.iloc[i]) != base_color:
        # No clean opposite-colored candle directly before the impulse.
        return None

    cluster_end = i

    # Walk back further collecting the full base cluster.
    while i >= 0 and _candle_color(df.iloc[i]) == base_color:
        i -= 1

    cluster_start = i + 1

    return cluster_start, cluster_end


def compute_zone_for_break(
    df: pd.DataFrame,
    brk: StructureBreak,
    break_idx: int,
) -> Optional[Zone]:
    """
    Incremental equivalent of one loop iteration inside find_zones().

    Computes the zone for exactly one structure break, given its
    already-known positional index in df, instead of rebuilding an
    index_pos lookup over the entire causal DataFrame on every call.

    A zone's origin cluster and boundaries are fixed the moment its
    break occurs (the walk in _find_origin_cluster is strictly
    backward from break_idx and never looks forward), so this is
    safe to call exactly once per new break and persist the result
    for the rest of the adapter's lifetime.

    find_zones() above remains completely untouched and is the
    correctness reference. This function must remain byte-for-byte
    equivalent to the per-break logic inside it -- see
    phase_02_optimization/test_zone_equivalence.py, which must pass
    before this function is used anywhere outside that test.
    """
    if df.empty:
        return None

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_index()

    impulse_color = "bullish" if brk.direction == Trend.BULLISH else "bearish"

    cluster = _find_origin_cluster(df, break_idx, impulse_color)
    if cluster is None:
        return None

    start_idx, end_idx = cluster
    cluster_slice = df.iloc[start_idx:end_idx + 1]

    zone_type = ZoneType.DEMAND if brk.direction == Trend.BULLISH else ZoneType.SUPPLY

    return Zone(
        zone_type=zone_type,
        price_top=cluster_slice["high"].max(),
        price_bottom=cluster_slice["low"].min(),
        origin_start=df.index[start_idx],
        origin_end=df.index[end_idx],
        formed_from_break_at=brk.broken_at,
    )


def find_zones(df: pd.DataFrame, breaks: List[StructureBreak]) -> List[Zone]:
    """
    Build Supply/Demand zones from confirmed structural breaks.
    Accepts the same raw OHLC DataFrame used for swings/structure,
    any column case (Open/High/Low/Close or lowercase).
    """
    if df.empty or not breaks:
        return []

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_index()

    index_pos = {ts: pos for pos, ts in enumerate(df.index)}

    zones: List[Zone] = []

    for brk in breaks:
        break_idx = index_pos.get(brk.broken_at)
        if break_idx is None:
            continue

        impulse_color = "bullish" if brk.direction == Trend.BULLISH else "bearish"

        cluster = _find_origin_cluster(df, break_idx, impulse_color)
        if cluster is None:
            continue

        start_idx, end_idx = cluster
        cluster_slice = df.iloc[start_idx:end_idx + 1]

        zone_type = ZoneType.DEMAND if brk.direction == Trend.BULLISH else ZoneType.SUPPLY

        zones.append(
            Zone(
                zone_type=zone_type,
                price_top=cluster_slice["high"].max(),
                price_bottom=cluster_slice["low"].min(),
                origin_start=df.index[start_idx],
                origin_end=df.index[end_idx],
                formed_from_break_at=brk.broken_at,
            )
        )

    return zones


def update_mitigation(zones: List[Zone], candle: pd.Series, ts: pd.Timestamp) -> None:
    """
    Mark any zone mitigated the instant this candle's wick overlaps its
    [price_bottom, price_top] range. Call once per candle, walking
    forward chronologically — a zone can't be mitigated by candles at
    or before the break that formed it.
    """
    for zone in zones:
        if zone.mitigated:
            continue
        if ts <= zone.formed_from_break_at:
            continue
        if candle["low"] <= zone.price_top and candle["high"] >= zone.price_bottom:
            zone.mitigated = True
            zone.mitigated_at = ts