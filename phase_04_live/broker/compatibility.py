
"""
phase_04_live/broker/compatibility.py

Phase 4 Step 2 -- Broker/Symbol Execution Compatibility
=========================================================

Purpose
-------
Resolve broker/symbol execution capabilities before OrderManager is
allowed to depend on a specific filling policy.

Core rule:

    NEVER GUESS BROKER CAPABILITIES.

This module is a read-only diagnostic and decision layer.

It does NOT:
    - place orders
    - modify broker state
    - modify strategy behavior
    - enable live trading
    - decide whether a trade signal should exist

Flow:

    mt5.symbol_info(symbol)
            |
            v
    decode_supported_filling_modes()
            |
            v
    select_order_filling_mode()
            |
            v
    check_symbol_compatibility()
            |
            v
    SymbolCompatibilityReport

The pure filling-policy functions are deterministic and can be tested
without a live MT5 connection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Optional

import MetaTrader5 as mt5


# ---------------------------------------------------------------------------
# MT5 symbol filling bit values
# ---------------------------------------------------------------------------
#
# The installed MetaTrader5 Python package exposes:
#
#     mt5.ORDER_FILLING_FOK
#     mt5.ORDER_FILLING_IOC
#
# but does NOT expose:
#
#     mt5.SYMBOL_FILLING_FOK
#     mt5.SYMBOL_FILLING_IOC
#
# symbol_info().filling_mode is a bitmask whose protocol values are:
#
#     FOK = 1
#     IOC = 2
#
# Keep these values explicit and local rather than referencing nonexistent
# package attributes.
#
# These are SYMBOL filling capability bits, NOT ORDER_FILLING_* request
# constants.
# ---------------------------------------------------------------------------

SYMBOL_FILLING_FOK = 1
SYMBOL_FILLING_IOC = 2


# ---------------------------------------------------------------------------
# Pure filling-mode logic
# ---------------------------------------------------------------------------


def decode_supported_filling_modes(
    filling_mode_bitmask: int,
) -> FrozenSet[str]:
    """
    Decode explicitly advertised FOK/IOC capabilities.

    Important:
        A zero bitmask does NOT mean "all filling modes supported".

    Unknown/reserved bits are ignored.

    RETURN is intentionally not decoded here because RETURN is governed
    by the symbol's execution mode rather than by a SYMBOL_FILLING_*
    capability bit.
    """
    modes = set()

    if filling_mode_bitmask & SYMBOL_FILLING_FOK:
        modes.add("FOK")

    if filling_mode_bitmask & SYMBOL_FILLING_IOC:
        modes.add("IOC")

    return frozenset(modes)


def return_filling_is_allowed(
    execution_mode: Optional[int],
) -> bool:
    """
    Determine whether ORDER_FILLING_RETURN is permitted.

    Fail closed for:
        - None
        - unknown execution modes
        - MARKET execution

    Permit RETURN for:
        - REQUEST
        - INSTANT
        - EXCHANGE
    """
    if execution_mode is None:
        return False

    if execution_mode == mt5.SYMBOL_TRADE_EXECUTION_MARKET:
        return False

    if execution_mode in (
        mt5.SYMBOL_TRADE_EXECUTION_REQUEST,
        mt5.SYMBOL_TRADE_EXECUTION_INSTANT,
        mt5.SYMBOL_TRADE_EXECUTION_EXCHANGE,
    ):
        return True

    return False


_PREFERENCE_ORDER = (
    "IOC",
    "FOK",
)


_ORDER_FILLING_CONSTANTS = {
    "FOK": mt5.ORDER_FILLING_FOK,
    "IOC": mt5.ORDER_FILLING_IOC,
    "RETURN": mt5.ORDER_FILLING_RETURN,
}


def select_order_filling_mode(
    filling_mode_bitmask: int,
    execution_mode: Optional[int] = None,
) -> int:
    """
    Resolve a safe MT5 ORDER_FILLING_* constant.

    Resolution order:

        1. Explicit IOC support
        2. Explicit FOK support
        3. RETURN when execution mode explicitly permits it
        4. RuntimeError / fail closed

    No broker capability is inferred.
    """
    supported = decode_supported_filling_modes(
        filling_mode_bitmask
    )

    # Explicit broker capability always takes priority.
    for candidate in _PREFERENCE_ORDER:
        if candidate in supported:
            return _ORDER_FILLING_CONSTANTS[candidate]

    # RETURN requires a known execution-mode policy.
    if return_filling_is_allowed(execution_mode):
        return _ORDER_FILLING_CONSTANTS["RETURN"]

    raise RuntimeError(
        "Cannot safely resolve order filling mode: "
        f"bitmask={filling_mode_bitmask!r} "
        f"and execution_mode={execution_mode!r} "
        "provide no valid filling option."
    )


# ---------------------------------------------------------------------------
# Human-readable MT5 names
# ---------------------------------------------------------------------------


_TRADE_MODE_NAMES = {
    0: "DISABLED",
    1: "LONGONLY",
    2: "SHORTONLY",
    3: "CLOSEONLY",
    4: "FULL",
}


_TRADE_EXECUTION_NAMES = {
    mt5.SYMBOL_TRADE_EXECUTION_REQUEST: "REQUEST",
    mt5.SYMBOL_TRADE_EXECUTION_INSTANT: "INSTANT",
    mt5.SYMBOL_TRADE_EXECUTION_MARKET: "MARKET",
    mt5.SYMBOL_TRADE_EXECUTION_EXCHANGE: "EXCHANGE",
}


def trade_execution_mode_name(
    execution_mode: Optional[int],
) -> Optional[str]:
    """Return a human-readable MT5 trade execution mode."""
    if execution_mode is None:
        return None

    return _TRADE_EXECUTION_NAMES.get(
        execution_mode,
        f"UNKNOWN({execution_mode})",
    )


def filling_mode_name(
    order_filling_mode: Optional[int],
) -> Optional[str]:
    """Return a human-readable ORDER_FILLING_* name."""
    if order_filling_mode is None:
        return None

    for name, value in _ORDER_FILLING_CONSTANTS.items():
        if value == order_filling_mode:
            return name

    return f"UNKNOWN({order_filling_mode})"


# ---------------------------------------------------------------------------
# Safe SymbolInfo extraction
# ---------------------------------------------------------------------------


def _get_execution_mode(info) -> Optional[int]:
    """
    Extract MT5 symbol execution mode.

    The standard Python MT5 SymbolInfo representation exposes the field
    as trade_exemode.

    Keep this isolated so connector-specific representation changes do
    not spread through the compatibility layer.
    """
    value = getattr(info, "trade_exemode", None)

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_numeric_field(
    info,
    field_name: str,
) -> Optional[float]:
    """
    Safely read a numeric SymbolInfo field.

    Returns None when the field does not exist or cannot be converted.
    """
    value = getattr(info, field_name, None)

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_int_field(
    info,
    field_name: str,
) -> Optional[int]:
    """
    Safely read an integer SymbolInfo field.

    Returns None when the field does not exist or cannot be converted.
    """
    value = getattr(info, field_name, None)

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Compatibility report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolCompatibilityReport:
    """
    Complete read-only compatibility report for one broker symbol.

    is_compatible is True only when:
        - the symbol exists
        - there are no blocking issues

    Warnings do not make the report incompatible by themselves.
    """

    symbol: str

    exists: bool
    visible: bool

    trade_mode: Optional[int]
    trade_mode_name: Optional[str]

    execution_mode: Optional[int]
    execution_mode_name: Optional[str]

    volume_min: Optional[float]
    volume_max: Optional[float]
    volume_step: Optional[float]

    point: Optional[float]
    digits: Optional[int]

    trade_stops_level: Optional[int]
    trade_freeze_level: Optional[int]

    raw_filling_mode_bitmask: Optional[int]
    supported_filling_modes: FrozenSet[str]

    resolved_order_filling_mode: Optional[int]
    resolved_order_filling_mode_name: Optional[str]

    blocking_issues: tuple = field(default_factory=tuple)
    warnings: tuple = field(default_factory=tuple)

    @property
    def is_compatible(self) -> bool:
        """Return True only when no blocking compatibility issue exists."""
        return self.exists and not self.blocking_issues


# ---------------------------------------------------------------------------
# Live compatibility check
# ---------------------------------------------------------------------------


def check_symbol_compatibility(
    symbol: str,
) -> SymbolCompatibilityReport:
    """
    Query MT5 and build a read-only compatibility report.

    Requirements:
        mt5.initialize() must already have succeeded.

    This function never sends or modifies an order.
    """
    info = mt5.symbol_info(symbol)

    if info is None:
        return SymbolCompatibilityReport(
            symbol=symbol,
            exists=False,
            visible=False,
            trade_mode=None,
            trade_mode_name=None,
            execution_mode=None,
            execution_mode_name=None,
            volume_min=None,
            volume_max=None,
            volume_step=None,
            point=None,
            digits=None,
            trade_stops_level=None,
            trade_freeze_level=None,
            raw_filling_mode_bitmask=None,
            supported_filling_modes=frozenset(),
            resolved_order_filling_mode=None,
            resolved_order_filling_mode_name=None,
            blocking_issues=(
                f"symbol_info() returned None for '{symbol}': "
                f"{mt5.last_error()}. "
                "Symbol may not exist on this broker/account or "
                "may need adding to Market Watch.",
            ),
            warnings=(),
        )

    blocking_issues = []
    warnings = []

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    visible = bool(getattr(info, "visible", False))

    if not visible:
        warnings.append(
            f"Symbol '{symbol}' is not currently visible in Market Watch."
        )

    # ------------------------------------------------------------------
    # Trade mode
    # ------------------------------------------------------------------

    trade_mode = _get_int_field(info, "trade_mode")

    if trade_mode is None:
        trade_mode_name = None
        blocking_issues.append(
            f"Symbol '{symbol}' does not expose a valid trade_mode."
        )
    else:
        trade_mode_name = _TRADE_MODE_NAMES.get(
            trade_mode,
            f"UNKNOWN({trade_mode})",
        )

        if trade_mode == 0:
            blocking_issues.append(
                f"Symbol '{symbol}' has trade_mode=DISABLED."
            )

        elif trade_mode == 3:
            blocking_issues.append(
                f"Symbol '{symbol}' has trade_mode=CLOSEONLY."
            )

        elif trade_mode == 1:
            warnings.append(
                f"Symbol '{symbol}' has trade_mode=LONGONLY; "
                "short signals cannot be opened."
            )

        elif trade_mode == 2:
            warnings.append(
                f"Symbol '{symbol}' has trade_mode=SHORTONLY; "
                "long signals cannot be opened."
            )

        elif trade_mode != 4:
            blocking_issues.append(
                f"Symbol '{symbol}' has unknown trade_mode="
                f"{trade_mode}."
            )

    # ------------------------------------------------------------------
    # Execution mode
    # ------------------------------------------------------------------

    execution_mode = _get_execution_mode(info)
    execution_mode_name = trade_execution_mode_name(
        execution_mode
    )

    if execution_mode is None:
        blocking_issues.append(
            f"Symbol '{symbol}' does not expose a valid trade execution "
            "mode; filling-policy compatibility cannot be resolved "
            "safely."
        )

    # ------------------------------------------------------------------
    # Volume constraints
    # ------------------------------------------------------------------

    volume_min = _get_numeric_field(info, "volume_min")
    volume_max = _get_numeric_field(info, "volume_max")
    volume_step = _get_numeric_field(info, "volume_step")

    if volume_min is None:
        blocking_issues.append(
            f"Symbol '{symbol}' does not expose volume_min."
        )
    elif volume_min <= 0:
        blocking_issues.append(
            f"Symbol '{symbol}' reports invalid volume_min="
            f"{volume_min}."
        )

    if volume_step is None:
        blocking_issues.append(
            f"Symbol '{symbol}' does not expose volume_step."
        )
    elif volume_step <= 0:
        blocking_issues.append(
            f"Symbol '{symbol}' reports invalid volume_step="
            f"{volume_step}."
        )

    if volume_max is None:
        blocking_issues.append(
            f"Symbol '{symbol}' does not expose volume_max."
        )
    elif volume_min is not None and volume_max < volume_min:
        blocking_issues.append(
            f"Symbol '{symbol}' reports volume_max "
            f"({volume_max}) < volume_min ({volume_min})."
        )

    # ------------------------------------------------------------------
    # Price precision
    # ------------------------------------------------------------------

    point = _get_numeric_field(info, "point")
    digits = _get_int_field(info, "digits")

    if point is None:
        blocking_issues.append(
            f"Symbol '{symbol}' does not expose a valid point value."
        )
    elif point <= 0:
        blocking_issues.append(
            f"Symbol '{symbol}' reports point={point}; "
            "point-based spread and price validation cannot "
            "function safely."
        )

    if digits is None:
        blocking_issues.append(
            f"Symbol '{symbol}' does not expose a valid digits value."
        )
    elif digits < 0:
        blocking_issues.append(
            f"Symbol '{symbol}' reports invalid digits={digits}."
        )

    # ------------------------------------------------------------------
    # Stop / freeze levels
    # ------------------------------------------------------------------

    trade_stops_level = _get_int_field(
        info,
        "trade_stops_level",
    )
    trade_freeze_level = _get_int_field(
        info,
        "trade_freeze_level",
    )

    if trade_stops_level is None:
        blocking_issues.append(
            f"Symbol '{symbol}' does not expose trade_stops_level."
        )
    elif trade_stops_level < 0:
        blocking_issues.append(
            f"Symbol '{symbol}' reports invalid "
            f"trade_stops_level={trade_stops_level}."
        )
    elif trade_stops_level > 0:
        warnings.append(
            f"Symbol '{symbol}' enforces a minimum stop distance of "
            f"{trade_stops_level} points. OrderManager currently "
            "does not validate this distance before submission."
        )

    if trade_freeze_level is None:
        blocking_issues.append(
            f"Symbol '{symbol}' does not expose trade_freeze_level."
        )
    elif trade_freeze_level < 0:
        blocking_issues.append(
            f"Symbol '{symbol}' reports invalid "
            f"trade_freeze_level={trade_freeze_level}."
        )
    elif trade_freeze_level > 0:
        warnings.append(
            f"Symbol '{symbol}' has a freeze level of "
            f"{trade_freeze_level} points. Position/SL/TP management "
            "must respect this broker constraint."
        )

    # ------------------------------------------------------------------
    # Filling mode
    # ------------------------------------------------------------------

    raw_filling_mode = _get_int_field(
        info,
        "filling_mode",
    )

    if raw_filling_mode is None:
        supported_filling_modes = frozenset()
        resolved_filling_mode = None

        blocking_issues.append(
            f"Symbol '{symbol}' does not expose a valid filling_mode "
            "bitmask; filling-policy compatibility cannot be resolved "
            "safely."
        )
    else:
        supported_filling_modes = decode_supported_filling_modes(
            raw_filling_mode
        )

        resolved_filling_mode = None

        try:
            resolved_filling_mode = select_order_filling_mode(
                raw_filling_mode,
                execution_mode=execution_mode,
            )
        except RuntimeError as exc:
            blocking_issues.append(str(exc))

    resolved_name = filling_mode_name(
        resolved_filling_mode
    )

    # ------------------------------------------------------------------
    # Existing OrderManager compatibility warning
    # ------------------------------------------------------------------

    if (
        resolved_filling_mode is not None
        and resolved_filling_mode != mt5.ORDER_FILLING_IOC
    ):
        warnings.append(
            f"Symbol '{symbol}' resolves to "
            f"ORDER_FILLING_{resolved_name}, while the current "
            "OrderManager hardcodes ORDER_FILLING_IOC. "
            "OrderManager must be updated before this symbol can "
            "be considered live-compatible."
        )

    elif (
        resolved_filling_mode is not None
        and "IOC" not in supported_filling_modes
    ):
        warnings.append(
            f"Symbol '{symbol}' does not explicitly report IOC "
            "support. The current OrderManager hardcodes "
            "ORDER_FILLING_IOC and therefore cannot safely trade "
            "this symbol until its filling policy is wired to the "
            "compatibility result."
        )

    return SymbolCompatibilityReport(
        symbol=symbol,
        exists=True,
        visible=visible,
        trade_mode=trade_mode,
        trade_mode_name=trade_mode_name,
        execution_mode=execution_mode,
        execution_mode_name=execution_mode_name,
        volume_min=volume_min,
        volume_max=volume_max,
        volume_step=volume_step,
        point=point,
        digits=digits,
        trade_stops_level=trade_stops_level,
        trade_freeze_level=trade_freeze_level,
        raw_filling_mode_bitmask=raw_filling_mode,
        supported_filling_modes=supported_filling_modes,
        resolved_order_filling_mode=resolved_filling_mode,
        resolved_order_filling_mode_name=resolved_name,
        blocking_issues=tuple(blocking_issues),
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------


def print_compatibility_report(
    report: SymbolCompatibilityReport,
) -> None:
    """Print a human-readable compatibility report."""
    print("=" * 70)
    print(
        f"SYMBOL COMPATIBILITY REPORT -- {report.symbol}"
    )
    print("=" * 70)

    if not report.exists:
        print("EXISTS                : No")

        for issue in report.blocking_issues:
            print(f"  [BLOCKING] {issue}")

        print("=" * 70)
        return

    print(f"exists                : {report.exists}")
    print(f"visible               : {report.visible}")
    print(f"trade_mode            : {report.trade_mode_name}")
    print(
        f"execution_mode        : "
        f"{report.execution_mode_name}"
    )
    print(
        f"volume_min/max/step   : "
        f"{report.volume_min} / "
        f"{report.volume_max} / "
        f"{report.volume_step}"
    )
    print(
        f"point / digits        : "
        f"{report.point} / {report.digits}"
    )
    print(
        f"trade_stops_level     : "
        f"{report.trade_stops_level} points"
    )
    print(
        f"trade_freeze_level    : "
        f"{report.trade_freeze_level} points"
    )
    print(
        f"raw filling bitmask   : "
        f"{report.raw_filling_mode_bitmask}"
    )
    print(
        f"supported filling     : "
        f"{sorted(report.supported_filling_modes)}"
    )
    print(
        f"resolved ORDER_FILLING: "
        f"{report.resolved_order_filling_mode_name} "
        f"(={report.resolved_order_filling_mode})"
    )

    if report.blocking_issues:
        print()
        print("BLOCKING ISSUES:")

        for issue in report.blocking_issues:
            print(f"  [BLOCKING] {issue}")

    if report.warnings:
        print()
        print("WARNINGS:")

        for warning in report.warnings:
            print(f"  [WARNING] {warning}")

    print()
    print(
        f"is_compatible: "
        f"{report.is_compatible}"
    )
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI diagnostic
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python -m phase_04_live.broker.compatibility "
            "SYMBOL [SYMBOL ...]"
        )
        print(
            "Requires a running MT5 terminal and successful "
            "mt5.initialize()."
        )
        sys.exit(1)

    if not mt5.initialize():
        print(
            f"mt5.initialize() failed: "
            f"{mt5.last_error()}"
        )
        sys.exit(1)

    try:
        for symbol in sys.argv[1:]:
            report = check_symbol_compatibility(symbol)
            print_compatibility_report(report)
            print()
    finally:
        mt5.shutdown()

