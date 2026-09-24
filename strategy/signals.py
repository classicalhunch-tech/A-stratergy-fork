"""
strategy/signals.py

Core signal generation engine for:
    - Liquidity sweeps
    - Structure breaks
    - Flip zones
    - Retest-triggered entries

Design goals:
    1. Preserve the existing strategy rules.
    2. Support compatibility / one-shot execution.
    3. Support optimized bar-by-bar backtesting.
    4. Avoid repeated dataframe preparation.
    5. Cache stable object -> bar-index resolutions.
    6. Preserve zone-consumption state across replay bars.
    7. Keep signal generation separate from forward retest simulation.
    8. Support detector objects that are recreated during replay.

RESEARCH METADATA (2026-09)
---------------------------
TradeSignal now carries `sweep_type` and `structure_break_type`. These
are NOT new classification logic -- they are the literal values that
this module already computes internally (via _resolve_level_type()
and _resolve_trend_token()) to decide LONG vs SHORT and to match a
structure break, but previously discarded once that decision was
made. Attaching them lets the Phase 1 research layer (see
strategy/backtest_report.py) study setup/mechanic behavior (BSL vs
SSL sweeps, BOS vs CHOCH vs INITIAL_BREAK) without this module or the
backtest engine changing behavior in any way.
"""

from dataclasses import dataclass
from enum import Enum
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Literal,
    Hashable,
)

import pandas as pd

from strategy.liquidity import LiquidityLevel, LiquidityStatus
from strategy.structure import StructureBreak
from strategy.zones import Zone


# =====================================================================
# ENUMS
# =====================================================================

class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class SignalStatus(Enum):
    PENDING_RETEST = "PENDING_RETEST"
    TRIGGERED = "TRIGGERED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


class EntryMode(Enum):
    MIDPOINT = "MIDPOINT"
    EXTREME = "EXTREME"


# =====================================================================
# CONSTANTS
# =====================================================================

_BULLISH_TOKENS = {
    "bullish",
    "long",
    "buy",
    "demand",
    "bos_bullish",
    "choch_bullish",
}

_BEARISH_TOKENS = {
    "bearish",
    "short",
    "sell",
    "supply",
    "bos_bearish",
    "choch_bearish",
}


# =====================================================================
# TRADE SIGNAL
# =====================================================================

@dataclass
class TradeSignal:
    """
    Immutable-style record of a generated trade setup.

    The object starts as PENDING_RETEST and may later become
    TRIGGERED or EXPIRED.

    `entry_price` can change during retest execution if the candle
    opening price provides a better fill.
    """

    signal_type: SignalType

    entry_price: float
    stop_loss: float
    take_profit: float

    entry_time: Optional[pd.Timestamp] = None
    exit_time: Optional[pd.Timestamp] = None

    risk_reward_ratio: float = 0.0

    status: SignalStatus = SignalStatus.PENDING_RETEST
    entry_mode: EntryMode = EntryMode.MIDPOINT

    zone_id: Optional[int] = None

    # Stable identity of the matched zone (see _stable_cache_key),
    # robust to zone objects being recreated across replay bars.
    # Callers that need to re-locate the originating zone later
    # (e.g. backtest.py re-matching a pending signal to its zone
    # on a subsequent bar) should key on this, not on zone_id
    # alone, since zone_id may be unset.
    zone_stable_key: Optional[Hashable] = None

    # Structure-break bar
    bar_index: int = 0

    # Liquidity-sweep bar
    setup_bar_index: Optional[int] = None

    # Liquidity-sweep timestamp
    setup_timestamp: Optional[pd.Timestamp] = None

    # Retest trigger metadata
    triggered_at_bar: Optional[int] = None
    triggered_at_time: Optional[pd.Timestamp] = None

    # Number of bars from structure break to retest trigger.
    bars_to_trigger: Optional[int] = None

    # Trading session assigned by the caller / session engine.
    session: str = "UNKNOWN"

    # ------------------------------------------------------------
    # RESEARCH METADATA (additive, non-functional)
    # ------------------------------------------------------------
    #
    # sweep_type: the normalized liquidity-side token this setup was
    # born from (e.g. "sell_side"/"ssl" -> LONG, "buy_side"/"bsl" ->
    # SHORT). This is exactly the `level_type` value already computed
    # by _resolve_level_type() in generate_flip_zone_signals() -- it
    # is only being *retained* here, not recomputed differently.
    sweep_type: Optional[str] = None

    # structure_break_type: the normalized trend/type token of the
    # confirming structure break (e.g. "bullish", "bos_bullish",
    # "choch_bearish" -- whatever _resolve_trend_token() resolves for
    # the real StructureBreak schema in strategy/structure.py). This
    # is the same value _find_matching_structure_break() already
    # computes internally to filter candidates; it is only being
    # retained here, not recomputed differently.
    structure_break_type: Optional[str] = None

    def __post_init__(self) -> None:
        self.recalculate_risk_reward()

    def recalculate_risk_reward(self) -> None:
        """
        Recalculate R:R from the current entry price.
        """

        risk = abs(
            self.entry_price - self.stop_loss
        )

        reward = abs(
            self.take_profit - self.entry_price
        )

        if risk > 0:
            self.risk_reward_ratio = (
                reward / risk
            )
        else:
            self.risk_reward_ratio = 0.0


# =====================================================================
# NORMALIZATION HELPERS
# =====================================================================

def _normalize_enum_or_value(value) -> str:
    """
    Normalize Enum/string-like values to lowercase token form.
    """

    if isinstance(value, Enum):
        value = value.value

    return (
        str(value)
        .split(".")[-1]
        .strip()
        .lower()
    )


def _resolve_level_type(
    liquidity: LiquidityLevel,
) -> str:
    """
    Resolve liquidity type across supported LiquidityLevel schemas.
    """

    raw_val = (
        getattr(liquidity, "type", None)
        or getattr(liquidity, "liquidity_type", None)
        or getattr(liquidity, "level_type", None)
        or getattr(liquidity, "side", None)
    )

    if raw_val is None:
        raise AttributeError(
            f"LiquidityLevel {liquidity!r} has no "
            "attribute type/side."
        )

    return _normalize_enum_or_value(
        raw_val
    )


def _resolve_trend_token(obj) -> str:
    """
    Resolve directional metadata across supported schemas.
    """

    raw_val = (
        getattr(obj, "trend", None)
        or getattr(obj, "zone_type", None)
        or getattr(obj, "structure_type", None)
        or getattr(obj, "direction", None)
        or getattr(obj, "type", None)
    )

    if raw_val is None:
        return ""

    return _normalize_enum_or_value(
        raw_val
    )


# =====================================================================
# CACHE KEY NORMALIZATION
# =====================================================================

def _make_hashable(value) -> Optional[Hashable]:
    """
    Convert common values into hashable representations.

    This prevents cache-key construction from failing when a detector
    stores an unusual but logically stable value.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool, bytes, tuple),
    ):
        try:
            hash(value)
            return value
        except TypeError:
            pass

    if isinstance(
        value,
        (pd.Timestamp, pd.Timedelta),
    ):
        return value

    try:
        return str(value)

    except Exception:
        return None


def _stable_cache_key(obj) -> Optional[Hashable]:
    """
    Build a stable content-based cache key.

    Detector objects may be recreated on every replay bar, so using
    id(obj) is unreliable for persistent replay caching.

    Priority:
        1. Explicit persistent identifiers.
        2. Combined historical identity fields.

    The key includes the object type so a Zone and StructureBreak
    sharing the same identifier do not collide.
    """

    if obj is None:
        return None

    object_type = type(obj).__name__

    # -------------------------------------------------------------
    # EXPLICIT STABLE IDENTIFIERS
    # -------------------------------------------------------------

    for attr in (
        "zone_id",
        "liquidity_id",
        "structure_id",
        "signal_id",
        "id",
    ):
        value = getattr(
            obj,
            attr,
            None,
        )

        if value is not None:
            hashable_value = _make_hashable(
                value
            )

            if hashable_value is not None:
                return (
                    object_type,
                    "identity",
                    attr,
                    hashable_value,
                )

    # -------------------------------------------------------------
    # COMPOSITE HISTORICAL IDENTITY
    # -------------------------------------------------------------

    identity_parts = []

    for attr in (
        "bar_index",
        "formed_bar_index",
        "break_bar_index",
        "confirmed_bar_index",
        "sweep_bar_index",
        "confirmed_at",
        "formed_at",
        "timestamp",
        "time",
        "swing_at",
        "broken_at",
        "origin_end",
        "swept_at",
        "formed_from_break_at",
    ):
        value = getattr(
            obj,
            attr,
            None,
        )

        if value is None:
            continue

        # Normalize timestamps so equivalent datetime objects
        # (pd.Timestamp, np.datetime64, datetime, etc.) produce
        # equivalent keys regardless of which concrete type the
        # detector happened to store.
        if (
            "at" in attr
            or attr in {
                "timestamp",
                "time",
                "origin_end",
            }
        ):
            try:
                value = pd.Timestamp(value)
            except (
                TypeError,
                ValueError,
            ):
                pass

        hashable_value = _make_hashable(
            value
        )

        if hashable_value is not None:
            identity_parts.append(
                (
                    attr,
                    hashable_value,
                )
            )

    if identity_parts:
        return (
            object_type,
            "historical",
            tuple(identity_parts),
        )

    return None


# =====================================================================
# DATAFRAME PREPARATION
# =====================================================================

def _prepare_working_dataframe(
    df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    str,
    Dict[pd.Timestamp, int],
]:
    """
    Prepare a dataframe for signal processing.

    Compatibility / one-shot path:

        - Copy dataframe
        - Normalize column names
        - Normalize timestamps
        - Sort chronologically
        - Build timestamp -> bar-index lookup
    """

    if df is None or df.empty:
        raise ValueError(
            "DataFrame cannot be None or empty."
        )

    working = df.copy()

    working.columns = [
        str(column).strip().lower()
        for column in working.columns
    ]

    if "timestamp" in working.columns:
        timestamp_column = "timestamp"

    elif isinstance(
        working.index,
        pd.DatetimeIndex,
    ):
        working = working.reset_index()

        working.columns = [
            str(column).strip().lower()
            for column in working.columns
        ]

        timestamp_column = (
            "timestamp"
            if "timestamp" in working.columns
            else working.columns[0]
        )

    else:
        raise ValueError(
            "DataFrame must contain a 'timestamp' "
            "column or DatetimeIndex."
        )

    required = {
        "open",
        "high",
        "low",
        "close",
    }

    missing = (
        required
        - set(working.columns)
    )

    if missing:
        raise ValueError(
            "DataFrame missing required columns: "
            f"{sorted(missing)}"
        )

    working[timestamp_column] = pd.to_datetime(
        working[timestamp_column],
        errors="coerce",
    )

    if working[timestamp_column].isna().any():
        raise ValueError(
            "DataFrame contains invalid timestamps."
        )

    working = (
        working
        .sort_values(
            timestamp_column,
            kind="stable",
        )
        .reset_index(drop=True)
    )

    timestamp_lookup: Dict[
        pd.Timestamp,
        int,
    ] = {}

    for idx, timestamp in enumerate(
        working[timestamp_column]
    ):
        if timestamp not in timestamp_lookup:
            timestamp_lookup[timestamp] = idx

    return (
        working,
        timestamp_column,
        timestamp_lookup,
    )


# =====================================================================
# FULL BACKTEST CONTEXT
# =====================================================================

def _prepare_full_context(
    df: pd.DataFrame,
) -> dict:
    """
    Prepare the complete historical dataframe once.

    Expensive operations are performed only once.

    Persistent state:
        - bar_index_cache
        - consumed_zones
    """

    (
        working,
        timestamp_column,
        timestamp_lookup,
    ) = _prepare_working_dataframe(df)

    return {
        "working": working,

        "timestamp_column":
            timestamp_column,

        "timestamp_lookup":
            timestamp_lookup,

        "highs":
            working["high"].to_numpy(),

        "lows":
            working["low"].to_numpy(),

        "opens":
            working["open"].to_numpy(),

        "timestamps":
            working[
                timestamp_column
            ].to_numpy(),

        # Persistent stable-object -> bar-index cache.
        "bar_index_cache": {},

        # Persistent logical-zone consumption state.
        "consumed_zones": set(),
    }


def _slice_prepared_context(
    precomputed_context: dict,
    upto_index: int,
) -> dict:
    """
    Return a cheap visible-history view.

    No dataframe copy, sorting, timestamp lookup rebuilding, or
    NumPy extraction occurs here.

    Persistent state is passed by reference.
    """

    if precomputed_context is None:
        raise ValueError(
            "precomputed_context cannot be None."
        )

    full_working = (
        precomputed_context["working"]
    )

    if len(full_working) == 0:
        raise ValueError(
            "Prepared dataframe cannot be empty."
        )

    if upto_index < 0:
        upto_index = 0

    if upto_index >= len(full_working):
        upto_index = (
            len(full_working) - 1
        )

    return {
        "working":
            full_working.iloc[
                :upto_index + 1
            ],

        "timestamp_column":
            precomputed_context[
                "timestamp_column"
            ],

        "timestamp_lookup":
            precomputed_context[
                "timestamp_lookup"
            ],

        "highs":
            precomputed_context["highs"],

        "lows":
            precomputed_context["lows"],

        "opens":
            precomputed_context["opens"],

        "timestamps":
            precomputed_context[
                "timestamps"
            ],

        "bar_index_cache":
            precomputed_context.get(
                "bar_index_cache"
            ),

        "consumed_zones":
            precomputed_context.get(
                "consumed_zones"
            ),
    }


# =====================================================================
# TIMESTAMP -> BAR INDEX
# =====================================================================

def _resolve_timestamp_to_bar_index(
    value,
    working: pd.DataFrame,
    timestamp_column: str,
    timestamp_lookup: Optional[
        Dict[pd.Timestamp, int]
    ] = None,
) -> Optional[int]:
    """
    Resolve a timestamp to its chronological bar index.

    Resolution order:
        1. Pre-built dictionary.
        2. DatetimeIndex lookup.
        3. Timestamp-column equality fallback.
    """

    if value is None:
        return None

    try:
        timestamp = pd.Timestamp(value)

    except (
        TypeError,
        ValueError,
    ):
        return None

    # Fast path.
    if timestamp_lookup is not None:
        return timestamp_lookup.get(
            timestamp
        )

    # DatetimeIndex fallback.
    if isinstance(
        working.index,
        pd.DatetimeIndex,
    ):
        try:
            location = (
                working.index.get_loc(
                    timestamp
                )
            )

            if isinstance(
                location,
                int,
            ):
                return location

            if isinstance(
                location,
                slice,
            ):
                return location.start

            if hasattr(
                location,
                "nonzero",
            ):
                positions = (
                    location.nonzero()[0]
                )

                if len(positions) > 0:
                    return int(
                        positions[0]
                    )

        except KeyError:
            pass

    # Timestamp-column fallback.
    if timestamp_column in working.columns:

        matches = working.index[
            working[timestamp_column]
            == timestamp
        ]

        if len(matches) > 0:
            return int(matches[0])

    return None


# =====================================================================
# BAR INDEX RESOLUTION
# =====================================================================

def _resolve_bar_index(
    obj,
    working: pd.DataFrame,
    timestamp_column: str,
    timestamp_lookup: Optional[
        Dict[pd.Timestamp, int]
    ] = None,
    bar_index_cache: Optional[dict] = None,
) -> Optional[int]:
    """
    Resolve a strategy object to its chronological bar index.

    Cached by stable logical identity rather than Python object identity.

    This is important because replay detectors may recreate equivalent
    objects on every bar.
    """

    if obj is None:
        return None

    # ================================================================
    # FAST CACHE PATH
    # ================================================================

    cache_key = None

    if bar_index_cache is not None:

        cache_key = _stable_cache_key(
            obj
        )

        if cache_key is not None:

            cached = bar_index_cache.get(
                cache_key
            )

            if cached is not None:
                return cached

    # ================================================================
    # DIRECT BAR-INDEX ATTRIBUTES
    # ================================================================

    for attr in (
        "bar_index",
        "formed_bar_index",
        "break_bar_index",
        "confirmed_bar_index",
        "sweep_bar_index",
    ):

        value = getattr(
            obj,
            attr,
            None,
        )

        if value is None:
            continue

        try:
            index = int(value)

        except (
            TypeError,
            ValueError,
        ):
            continue

        if 0 <= index < len(working):

            if (
                bar_index_cache is not None
                and cache_key is not None
            ):
                bar_index_cache[
                    cache_key
                ] = index

            return index

    # ================================================================
    # TIMESTAMP ATTRIBUTES
    # ================================================================

    for attr in (
        "confirmed_at",
        "formed_at",
        "timestamp",
        "time",
        "swing_at",
        "broken_at",
        "origin_end",
        "swept_at",
        "formed_from_break_at",
    ):

        value = getattr(
            obj,
            attr,
            None,
        )

        if value is None:
            continue

        index = (
            _resolve_timestamp_to_bar_index(
                value=value,
                working=working,
                timestamp_column=timestamp_column,
                timestamp_lookup=timestamp_lookup,
            )
        )

        if index is not None:

            if (
                bar_index_cache is not None
                and cache_key is not None
            ):
                bar_index_cache[
                    cache_key
                ] = index

            return index

    return None


# =====================================================================
# STRUCTURE BREAK MATCHING
# =====================================================================

def _find_matching_structure_break(
    sweep_idx: int,
    level_type: str,
    structure_breaks: List[StructureBreak],
    working: pd.DataFrame,
    timestamp_column: str,
    max_bars_after_sweep: int,
    timestamp_lookup: Optional[
        Dict[pd.Timestamp, int]
    ] = None,
    bar_index_cache: Optional[dict] = None,
) -> Optional[StructureBreak]:
    """
    Find the earliest valid structure break after a liquidity sweep.
    """

    matching_breaks = []

    for structure_break in structure_breaks:

        break_idx = _resolve_bar_index(
            structure_break,
            working,
            timestamp_column,
            timestamp_lookup,
            bar_index_cache,
        )

        if break_idx is None:
            continue

        # Break must occur strictly after sweep.
        if not (
            sweep_idx
            < break_idx
            <= (
                sweep_idx
                + max_bars_after_sweep
            )
        ):
            continue

        trend_normalized = (
            _resolve_trend_token(
                structure_break
            )
        )

        # SSL / sell-side sweep -> bullish confirmation.
        if (
            level_type
            in (
                "sell_side",
                "ssl",
                "low",
            )
            and trend_normalized
            in _BULLISH_TOKENS
        ):
            matching_breaks.append(
                (
                    break_idx,
                    structure_break,
                )
            )

        # BSL / buy-side sweep -> bearish confirmation.
        elif (
            level_type
            in (
                "buy_side",
                "bsl",
                "high",
            )
            and trend_normalized
            in _BEARISH_TOKENS
        ):
            matching_breaks.append(
                (
                    break_idx,
                    structure_break,
                )
            )

    if not matching_breaks:
        return None

    # Earliest valid break wins.
    matching_breaks.sort(
        key=lambda item: item[0]
    )

    return matching_breaks[0][1]


# =====================================================================
# FLIP ZONE MATCHING
# =====================================================================

def _zone_consumption_keys(
    zone: Zone,
) -> Set[Hashable]:
    """
    Build the persistent keys used to identify a logical zone.

    Explicit zone_id is preferred. The stable cache key is included
    as a fallback / secondary identity.
    """

    keys: Set[Hashable] = set()

    zone_id = getattr(
        zone,
        "zone_id",
        None,
    )

    if zone_id is not None:
        keys.add(zone_id)

    stable_key = _stable_cache_key(
        zone
    )

    if stable_key is not None:
        keys.add(stable_key)

    return keys


def _is_zone_consumed(
    zone: Zone,
    consumed_zones: Set[Hashable],
) -> bool:
    """
    Determine whether the logical zone has already been consumed.
    """

    for key in _zone_consumption_keys(
        zone
    ):
        if key in consumed_zones:
            return True

    return False


def _consume_zone(
    zone: Zone,
    consumed_zones: Set[Hashable],
) -> None:
    """
    Persistently mark a logical zone as consumed.
    """

    for key in _zone_consumption_keys(
        zone
    ):
        consumed_zones.add(key)


def _find_matching_flip_zone(
    signal_type: SignalType,
    structure_break: StructureBreak,
    zones: List[Zone],
    working: pd.DataFrame,
    timestamp_column: str,
    max_bars_from_break: int,
    consumed_zones: Set[Hashable],
    timestamp_lookup: Optional[
        Dict[pd.Timestamp, int]
    ] = None,
    bar_index_cache: Optional[dict] = None,
) -> Optional[Zone]:
    """
    Find the most recent qualifying unconsumed flip zone.

    Rejection conditions:
        - Already consumed
        - Mitigated
        - Wrong direction
        - Formed after structure break
        - Too far from structure break
    """

    break_idx = _resolve_bar_index(
        structure_break,
        working,
        timestamp_column,
        timestamp_lookup,
        bar_index_cache,
    )

    if break_idx is None:
        return None

    candidates = []

    for zone in zones:

        # -------------------------------------------------------------
        # CONSUMPTION
        # -------------------------------------------------------------

        if _is_zone_consumed(
            zone,
            consumed_zones,
        ):
            continue

        # -------------------------------------------------------------
        # MITIGATION
        # -------------------------------------------------------------

        is_mitigated = (
            getattr(
                zone,
                "is_mitigated",
                False,
            )
            or getattr(
                zone,
                "mitigated",
                False,
            )
        )

        if is_mitigated:
            continue

        # -------------------------------------------------------------
        # DIRECTION
        # -------------------------------------------------------------

        zone_trend = (
            _resolve_trend_token(zone)
        )

        if (
            signal_type == SignalType.LONG
            and zone_trend
            not in _BULLISH_TOKENS
        ):
            continue

        if (
            signal_type == SignalType.SHORT
            and zone_trend
            not in _BEARISH_TOKENS
        ):
            continue

        # -------------------------------------------------------------
        # FORMATION INDEX
        # -------------------------------------------------------------

        zone_idx = _resolve_bar_index(
            zone,
            working,
            timestamp_column,
            timestamp_lookup,
            bar_index_cache,
        )

        if zone_idx is None:
            continue

        # Zone must exist no later than break.
        if zone_idx > break_idx:
            continue

        # Zone must be within configured distance.
        if (
            break_idx
            - zone_idx
            > max_bars_from_break
        ):
            continue

        candidates.append(
            (
                zone_idx,
                zone,
            )
        )

    if not candidates:
        return None

    # Most recently formed qualifying zone wins.
    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return candidates[0][1]


# =====================================================================
# SIGNAL GENERATION
# =====================================================================

def generate_flip_zone_signals(
    liquidity_levels: List[LiquidityLevel],
    structure_breaks: List[StructureBreak],
    zones: List[Zone],
    df: Optional[pd.DataFrame] = None,

    max_bars_after_sweep: int = 15,
    max_bars_from_break: int = 5,
    max_bars_to_retest: int = 50,

    reward_multiple: float = 2.5,
    stop_buffer: float = 0.0,

    entry_mode: Literal[
        "midpoint",
        "extreme",
    ] = "midpoint",

    precomputed_context: Optional[dict] = None,
    upto_index: Optional[int] = None,

    run_retest_simulation: bool = True,

    consumed_zones: Optional[
        Set[Hashable]
    ] = None,

    session: str = "UNKNOWN",
) -> List[TradeSignal]:
    """
    Generate Flip Zone trade signals.

    COMPATIBILITY / ONE-SHOT PATH
    ------------------------------
    Uses df directly and prepares the dataframe once.

    OPTIMIZED BAR-BY-BAR PATH
    --------------------------
    Uses precomputed_context + upto_index.

    RETEST MODES
    ------------
    run_retest_simulation=True
        This function may scan forward through candles visible
        in the supplied working dataframe.

    run_retest_simulation=False
        This function returns fresh PENDING_RETEST signals.
        The caller, normally backtest.py, owns the future-bar
        retest state machine.

    ZONE STATE
    ----------
    consumed_zones may be supplied externally so that logical
    zones remain consumed across replay bars even when detector
    objects are recreated.
    """

    # =================================================================
    # PARAMETER VALIDATION
    # =================================================================

    if max_bars_after_sweep <= 0:
        raise ValueError(
            "max_bars_after_sweep must be > 0"
        )

    if max_bars_from_break < 0:
        raise ValueError(
            "max_bars_from_break cannot be negative"
        )

    if max_bars_to_retest <= 0:
        raise ValueError(
            "max_bars_to_retest must be > 0"
        )

    if reward_multiple <= 0:
        raise ValueError(
            "reward_multiple must be > 0"
        )

    if stop_buffer < 0:
        raise ValueError(
            "stop_buffer cannot be negative"
        )

    if entry_mode not in (
        "midpoint",
        "extreme",
    ):
        raise ValueError(
            "entry_mode must be "
            "'midpoint' or 'extreme'."
        )

    # Extreme mode requires a non-zero stop distance.
    if (
        entry_mode == "extreme"
        and stop_buffer <= 0
    ):
        stop_buffer = 0.01

    # =================================================================
    # EMPTY INPUT GUARD
    # =================================================================

    if (
        not liquidity_levels
        or not structure_breaks
        or not zones
    ):
        return []

    # =================================================================
    # PREPARE DATA
    # =================================================================

    if precomputed_context is not None:

        if upto_index is None:

            if df is None or df.empty:
                return []

            upto_index = len(df) - 1

        sliced = _slice_prepared_context(
            precomputed_context,
            upto_index,
        )

        working = sliced["working"]

        timestamp_column = (
            sliced["timestamp_column"]
        )

        timestamp_lookup = (
            sliced["timestamp_lookup"]
        )

        highs = sliced["highs"]
        lows = sliced["lows"]
        opens = sliced["opens"]
        timestamps = sliced["timestamps"]

        bar_index_cache = (
            sliced["bar_index_cache"]
        )

        if consumed_zones is None:
            consumed_zones = (
                sliced["consumed_zones"]
            )

    else:

        if df is None or df.empty:
            return []

        (
            working,
            timestamp_column,
            timestamp_lookup,
        ) = _prepare_working_dataframe(
            df
        )

        highs = (
            working["high"].to_numpy()
        )

        lows = (
            working["low"].to_numpy()
        )

        opens = (
            working["open"].to_numpy()
        )

        timestamps = (
            working[
                timestamp_column
            ].to_numpy()
        )

        # One-shot compatibility cache.
        bar_index_cache = {}

    # Compatibility fallback.
    if consumed_zones is None:
        consumed_zones = set()

    signals: List[TradeSignal] = []

    # =================================================================
    # COLLECT SWEPT LIQUIDITY
    # =================================================================

    swept_liquidity = []

    for level in liquidity_levels:

        status = getattr(
            level,
            "status",
            None,
        )

        status_val = str(
            status
        ).upper()

        is_swept = (
            status_val == "SWEPT"
            or status == LiquidityStatus.SWEPT
        )

        if not is_swept:
            continue

        sweep_idx = getattr(
            level,
            "sweep_bar_index",
            None,
        )

        if sweep_idx is None:

            sweep_idx = (
                _resolve_bar_index(
                    level,
                    working,
                    timestamp_column,
                    timestamp_lookup,
                    bar_index_cache,
                )
            )

        if sweep_idx is not None:
            swept_liquidity.append(
                (
                    int(sweep_idx),
                    level,
                )
            )

    if not swept_liquidity:
        return []

    swept_liquidity.sort(
        key=lambda item: item[0]
    )

    # =================================================================
    # PROCESS EACH LIQUIDITY SWEEP
    # =================================================================

    for (
        sweep_idx,
        liquidity,
    ) in swept_liquidity:

        if not (
            0 <= sweep_idx < len(working)
        ):
            continue

        # -------------------------------------------------------------
        # SIGNAL DIRECTION
        # -------------------------------------------------------------

        level_type = (
            _resolve_level_type(
                liquidity
            )
        )

        if level_type in (
            "sell_side",
            "ssl",
            "low",
        ):
            signal_type = SignalType.LONG

        elif level_type in (
            "buy_side",
            "bsl",
            "high",
        ):
            signal_type = SignalType.SHORT

        else:
            continue

        # -------------------------------------------------------------
        # STRUCTURE BREAK
        # -------------------------------------------------------------

        matching_break = (
            _find_matching_structure_break(
                sweep_idx=sweep_idx,
                level_type=level_type,
                structure_breaks=structure_breaks,
                working=working,
                timestamp_column=timestamp_column,
                max_bars_after_sweep=(
                    max_bars_after_sweep
                ),
                timestamp_lookup=timestamp_lookup,
                bar_index_cache=bar_index_cache,
            )
        )

        if matching_break is None:
            continue

        # -------------------------------------------------------------
        # RESEARCH METADATA -- reuses the SAME classification logic
        # that was already used above to accept this sweep/break pair.
        # No new detection rule is introduced here.
        # -------------------------------------------------------------

        resolved_structure_break_type = (
            _resolve_trend_token(
                matching_break
            )
        )

        # -------------------------------------------------------------
        # FLIP ZONE
        # -------------------------------------------------------------

        matching_zone = (
            _find_matching_flip_zone(
                signal_type=signal_type,
                structure_break=matching_break,
                zones=zones,
                working=working,
                timestamp_column=timestamp_column,
                max_bars_from_break=(
                    max_bars_from_break
                ),
                consumed_zones=consumed_zones,
                timestamp_lookup=timestamp_lookup,
                bar_index_cache=bar_index_cache,
            )
        )

        if matching_zone is None:
            continue

        # -------------------------------------------------------------
        # ZONE IDENTIFICATION
        # -------------------------------------------------------------

        zone_id_attr = getattr(
            matching_zone,
            "zone_id",
            None,
        )

        resolved_zone_id = zone_id_attr

        # Stable identity of this zone, robust to the zone object
        # being recreated on a later replay bar (see backtest.py's
        # _visible_zones, which returns a fresh copy every call).
        resolved_zone_stable_key = _stable_cache_key(
            matching_zone
        )

        # -------------------------------------------------------------
        # ZONE BOUNDARIES
        # -------------------------------------------------------------

        try:

            price_top = float(
                matching_zone.price_top
            )

            price_bottom = float(
                matching_zone.price_bottom
            )

        except (
            AttributeError,
            TypeError,
            ValueError,
        ):
            continue

        if price_top <= price_bottom:
            continue

        # -------------------------------------------------------------
        # TRADE LEVELS
        # -------------------------------------------------------------

        if signal_type == SignalType.LONG:

            stop_loss = (
                price_bottom
                - stop_buffer
            )

            if entry_mode == "extreme":
                entry_price = price_bottom

            else:
                entry_price = (
                    price_top
                    + price_bottom
                ) / 2.0

            risk = (
                entry_price
                - stop_loss
            )

            if risk <= 0:
                continue

            take_profit = (
                entry_price
                + risk * reward_multiple
            )

        else:

            stop_loss = (
                price_top
                + stop_buffer
            )

            if entry_mode == "extreme":
                entry_price = price_top

            else:
                entry_price = (
                    price_top
                    + price_bottom
                ) / 2.0

            risk = (
                stop_loss
                - entry_price
            )

            if risk <= 0:
                continue

            take_profit = (
                entry_price
                - risk * reward_multiple
            )

        # -------------------------------------------------------------
        # STRUCTURE BREAK INDEX
        # -------------------------------------------------------------

        break_idx = _resolve_bar_index(
            matching_break,
            working,
            timestamp_column,
            timestamp_lookup,
            bar_index_cache,
        )

        if break_idx is None:
            continue

        signal_timestamp = pd.Timestamp(
            timestamps[break_idx]
        )

        setup_timestamp = pd.Timestamp(
            timestamps[sweep_idx]
        )

        # -------------------------------------------------------------
        # CREATE SIGNAL
        # -------------------------------------------------------------

        trade_signal = TradeSignal(
            signal_type=signal_type,

            entry_price=float(
                entry_price
            ),

            stop_loss=float(
                stop_loss
            ),

            take_profit=float(
                take_profit
            ),

            entry_time=signal_timestamp,

            status=(
                SignalStatus.PENDING_RETEST
            ),

            entry_mode=(
                EntryMode.MIDPOINT
                if entry_mode == "midpoint"
                else EntryMode.EXTREME
            ),

            zone_id=resolved_zone_id,

            zone_stable_key=resolved_zone_stable_key,

            bar_index=int(
                break_idx
            ),

            setup_bar_index=int(
                sweep_idx
            ),

            setup_timestamp=setup_timestamp,

            session=str(
                session
            ),

            sweep_type=level_type,

            structure_break_type=(
                resolved_structure_break_type
                or None
            ),
        )

        # =================================================================
        # OPTIONAL FORWARD RETEST
        # =================================================================

        if run_retest_simulation:

            retested = False

            end_idx = min(
                break_idx
                + max_bars_to_retest
                + 1,
                len(working),
            )

            for idx in range(
                break_idx + 1,
                end_idx,
            ):

                bar_high = float(
                    highs[idx]
                )

                bar_low = float(
                    lows[idx]
                )

                bar_open = float(
                    opens[idx]
                )

                bar_time = pd.Timestamp(
                    timestamps[idx]
                )

                # ---------------------------------------------------------
                # LONG RETEST
                # ---------------------------------------------------------

                if (
                    signal_type
                    == SignalType.LONG
                ):

                    if bar_low <= price_top:

                        trade_signal.status = (
                            SignalStatus.TRIGGERED
                        )

                        trade_signal.triggered_at_bar = (
                            int(idx)
                        )

                        trade_signal.triggered_at_time = (
                            bar_time
                        )

                        trade_signal.bars_to_trigger = (
                            int(idx - break_idx)
                        )

                        # Better long fill at candle open.
                        if (
                            bar_open
                            < trade_signal.entry_price
                        ):
                            trade_signal.entry_price = (
                                bar_open
                            )

                        trade_signal.entry_time = (
                            bar_time
                        )

                        # Entry changed -> update R:R.
                        trade_signal.recalculate_risk_reward()

                        retested = True
                        break

                # ---------------------------------------------------------
                # SHORT RETEST
                # ---------------------------------------------------------

                else:

                    if bar_high >= price_bottom:

                        trade_signal.status = (
                            SignalStatus.TRIGGERED
                        )

                        trade_signal.triggered_at_bar = (
                            int(idx)
                        )

                        trade_signal.triggered_at_time = (
                            bar_time
                        )

                        trade_signal.bars_to_trigger = (
                            int(idx - break_idx)
                        )

                        # Better short fill at candle open.
                        if (
                            bar_open
                            > trade_signal.entry_price
                        ):
                            trade_signal.entry_price = (
                                bar_open
                            )

                        trade_signal.entry_time = (
                            bar_time
                        )

                        # Entry changed -> update R:R.
                        trade_signal.recalculate_risk_reward()

                        retested = True
                        break

            # -------------------------------------------------------------
            # RETEST EXPIRED
            # -------------------------------------------------------------

            if not retested:

                trade_signal.status = (
                    SignalStatus.EXPIRED
                )

        # =================================================================
        # CONSUME ZONE AFTER VALID SIGNAL CONSTRUCTION
        # =================================================================
        #
        # IMPORTANT:
        #
        # Consumption happens only after:
        #     - zone boundaries are valid
        #     - risk is valid
        #     - break index is valid
        #     - TradeSignal has been successfully created
        #
        # The logical identity is persisted through _stable_cache_key()
        # and/or the explicit zone_id, so recreated detector objects
        # cannot bypass consumption.
        # =================================================================

        _consume_zone(
            matching_zone,
            consumed_zones,
        )

        # -------------------------------------------------------------
        # STORE SIGNAL
        # -------------------------------------------------------------

        signals.append(
            trade_signal
        )

    # =================================================================
    # CHRONOLOGICAL OUTPUT
    # =================================================================

    signals.sort(
        key=lambda signal: signal.bar_index
    )

    return signals