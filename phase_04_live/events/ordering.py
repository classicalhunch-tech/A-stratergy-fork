"""
phase_04_live/events/ordering.py

Deterministic ordering for one live-loop cycle's events.

Fixed priority, most operationally urgent first:
    1. Clock issues (drift/staleness) -- the whole cycle's data may be
       untrustworthy if these fire, so they surface first.
    2. Quality issues -- bad data was already dropped, but should be
       visible before anything else this cycle.
    3. Tick/adapter errors -- something broke.
    4. Suppressed signals -- a signal existed but was correctly withheld.
    5. Triggered signals -- the actual trading outcome of the cycle.

Within the same type, original chronological order is preserved
(Python's sort is stable).
"""

from typing import List

from phase_04_live.events.event_types import LiveEvent, LiveEventType


_PRIORITY = {
    LiveEventType.CLOCK_DRIFT: 0,
    LiveEventType.CLOCK_STALE: 0,
    LiveEventType.QUALITY_ISSUE: 1,
    LiveEventType.TICK_ERROR: 2,
    LiveEventType.ADAPTER_ERROR: 2,
    LiveEventType.SIGNAL_SUPPRESSED: 3,
    LiveEventType.SIGNAL_TRIGGERED: 4,
}


def order_events(events: List[LiveEvent]) -> List[LiveEvent]:
    """Return events sorted by fixed priority, stable within a type."""
    return sorted(events, key=lambda e: _PRIORITY.get(e.event_type, 99))
