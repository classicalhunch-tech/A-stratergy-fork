"""
phase_04_live/clock/models.py

Data model for clock integrity issues.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ClockIssueType(str, Enum):
    CLOCK_DRIFT = "CLOCK_DRIFT"
    FEED_STALE = "FEED_STALE"


@dataclass(frozen=True)
class ClockIssue:
    issue_type: ClockIssueType
    detail: str
    measured_seconds: float
    threshold_seconds: float
    detected_at: datetime
