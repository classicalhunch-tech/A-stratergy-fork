"""
phase_04_live/events/event_types.py

Unified event shape for everything the Phase 4 live loop surfaces in
one cycle: clock issues, market quality issues, execution results, and
whatever RuntimeCoordinator already produces (triggered/suppressed signals,
adapter errors, tick errors).

This does NOT replace RuntimeCoordinator's own event handling (queued
triggered events, audit log) -- it wraps those alongside the
phase_04_live-specific event types (clock/quality/execution) that
RuntimeCoordinator has no knowledge of, so monitoring/ has one
consistent stream to read from instead of three separate sources.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class LiveEventType(str, Enum):
    CLOCK_DRIFT = "CLOCK_DRIFT"
    CLOCK_STALE = "CLOCK_STALE"
    QUALITY_ISSUE = "QUALITY_ISSUE"
    TICK_ERROR = "TICK_ERROR"
    ADAPTER_ERROR = "ADAPTER_ERROR"
    SIGNAL_TRIGGERED = "SIGNAL_TRIGGERED"
    SIGNAL_SUPPRESSED = "SIGNAL_SUPPRESSED"
    ORDER_PLACED = "ORDER_PLACED"
    ORDER_REJECTED = "ORDER_REJECTED"
    ORDER_ERROR = "ORDER_ERROR"


@dataclass(frozen=True)
class LiveEvent:
    """
    One normalized event for the live loop's output stream.

    payload holds the original object (ClockIssue, QualityIssue,
    AdapterSignalEvent, ExecutionResult, or a raw error string) --
    LiveEvent itself is just a consistent envelope, it does not replace
    or reinterpret the original data.
    """
    event_type: LiveEventType
    occurred_at: datetime
    payload: Any
