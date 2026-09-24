"""
strategy/silence_diagnostics.py

Explains trading silence instead of leaving it unexplained.

Problem this solves
--------------------
The MTF confluence gate can correctly go quiet for extended real
periods (see: 2026-03-15 to 2026-05-20 dead zone during a strong
Gold uptrend, where SHORT signals were almost entirely and
correctly rejected for disagreeing with a bullish macro trend).

That silence is CORRECT behavior, not a bug -- but in live trading,
silence with no explanation is indistinguishable from a broken
system. This module makes the reason visible.

It does not change any trading logic. It only reports on decisions
the MTF filter already made.
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

from strategy.confluence import build_mtf_signal_filter


@dataclass
class RejectionRecord:
    setup_time: pd.Timestamp
    direction: str
    macro_trend: Optional[str]
    internal_trend: Optional[str]
    approved: bool


@dataclass
class SilenceReport:
    """Summary of why signals were or weren't approved over a period."""

    period_start: pd.Timestamp
    period_end: pd.Timestamp

    total_signals: int
    total_approved: int
    total_rejected: int

    # Direction breakdown
    long_signals: int
    long_approved: int
    short_signals: int
    short_approved: int

    # Rejection reason breakdown: counts of (direction, macro_trend, internal_trend)
    # combinations among REJECTED signals only.
    rejection_breakdown: dict

    def summary_lines(self) -> List[str]:
        lines = []
        lines.append(
            f"Period: {self.period_start} -> {self.period_end}"
        )
        lines.append(
            f"Signals: {self.total_signals} total "
            f"({self.total_approved} approved, {self.total_rejected} rejected)"
        )

        if self.long_signals:
            long_rate = self.long_approved / self.long_signals * 100.0
        else:
            long_rate = 0.0

        if self.short_signals:
            short_rate = self.short_approved / self.short_signals * 100.0
        else:
            short_rate = 0.0

        lines.append(
            f"  LONG:  {self.long_approved}/{self.long_signals} approved "
            f"({long_rate:.1f}%)"
        )
        lines.append(
            f"  SHORT: {self.short_approved}/{self.short_signals} approved "
            f"({short_rate:.1f}%)"
        )

        if self.total_rejected > 0:
            lines.append("Top rejection reasons (direction, macro, internal):")
            sorted_reasons = sorted(
                self.rejection_breakdown.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )
            for (direction, macro, internal), count in sorted_reasons[:5]:
                lines.append(
                    f"  {direction} rejected {count}x "
                    f"(macro={macro}, internal={internal})"
                )

        if self.total_signals == 0:
            lines.append(
                "  NOTE: zero raw signals generated in this period -- "
                "this is NOT an MTF rejection issue, the base 5M engine "
                "found no candidate setups at all. Check volatility / "
                "structure activity for this period separately."
            )
        elif self.total_approved == 0:
            # Identify dominant one-sided trend as likely explanation.
            long_blocked = self.long_signals > 0 and self.long_approved == 0
            short_blocked = self.short_signals > 0 and self.short_approved == 0
            if long_blocked and short_blocked:
                lines.append(
                    "  NOTE: both directions fully blocked -- likely a "
                    "strongly one-sided or choppy higher-timeframe trend "
                    "disagreeing with internal structure. This can be "
                    "correct behavior during strong trends; verify against "
                    "macro/internal trend distribution before assuming a bug."
                )

        return lines


def build_silence_report(
    df_slice: pd.DataFrame,
    raw_signals: list,
    df_enriched: pd.DataFrame,
    allow_neutral_internal: bool = True,
) -> SilenceReport:
    """
    Build a SilenceReport explaining MTF approval/rejection behavior
    over df_slice's period, given the raw (ungated) signals that were
    generated and the MTF-enriched context.

    Parameters
    ----------
    df_slice:
        The 5M OHLC slice being reported on (used only for its time bounds).
    raw_signals:
        Output of generate_flip_zone_signals() -- the UNGATED candidate
        signals for this period.
    df_enriched:
        Output of build_mtf_dataset_with_structure() for the SAME
        underlying dataset the signals were generated from.
    """

    mtf_filter_fn = build_mtf_signal_filter(
        df_enriched,
        allow_neutral_internal=allow_neutral_internal,
    )

    records: List[RejectionRecord] = []

    for signal in raw_signals:
        setup_time = getattr(signal, "setup_timestamp", None)
        if setup_time is None:
            continue
        setup_time = pd.Timestamp(setup_time)

        direction = getattr(signal.signal_type, "value", signal.signal_type)
        direction = str(direction).upper()

        approved = mtf_filter_fn(signal, setup_time)

        macro = df_enriched["macro_trend"].asof(setup_time)
        internal = df_enriched["internal_trend"].asof(setup_time)

        records.append(
            RejectionRecord(
                setup_time=setup_time,
                direction=direction,
                macro_trend=macro,
                internal_trend=internal,
                approved=approved,
            )
        )

    total = len(records)
    approved_records = [r for r in records if r.approved]
    rejected_records = [r for r in records if not r.approved]

    long_records = [r for r in records if r.direction == "LONG"]
    short_records = [r for r in records if r.direction == "SHORT"]

    rejection_breakdown: dict = {}
    for r in rejected_records:
        key = (r.direction, r.macro_trend, r.internal_trend)
        rejection_breakdown[key] = rejection_breakdown.get(key, 0) + 1

    return SilenceReport(
        period_start=df_slice.index[0] if len(df_slice) else pd.NaT,
        period_end=df_slice.index[-1] if len(df_slice) else pd.NaT,
        total_signals=total,
        total_approved=len(approved_records),
        total_rejected=len(rejected_records),
        long_signals=len(long_records),
        long_approved=sum(1 for r in long_records if r.approved),
        short_signals=len(short_records),
        short_approved=sum(1 for r in short_records if r.approved),
        rejection_breakdown=rejection_breakdown,
    )


def print_silence_report(report: SilenceReport) -> None:
    """Print a readable silence-diagnostic report."""
    print("-" * 70)
    print("SILENCE DIAGNOSTIC REPORT")
    print("-" * 70)
    for line in report.summary_lines():
        print(line)
    print("-" * 70)