"""
phase_04_live/risk/live_metrics.py

Real daily-loss (in R) and drawdown measurements for populating
RiskGateInput.daily_realized_r and current_drawdown_pct.

Daily R chain (per user's architecture):
    MT5 closed deals -> per-trade realized profit -> each trade's
    recorded initial dollar risk (Phase4Persistence.trade_risk,
    written from PositionSizeResult.dollar_risk at open) -> R multiple
    -> summed across today -> daily realized R.

A closed deal whose original trade_risk was never recorded (e.g. a
manual trade placed outside this loop) is skipped, not guessed --
silently inventing a risk denominator would produce a fabricated R
value, which is worse than omitting that trade from the sum.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5

from phase_04_live.persistence.store import Phase4Persistence


def get_daily_realized_r(persistence: Phase4Persistence, symbol: str) -> float:
    """
    Sum of R-multiples across every closing deal on `symbol` since
    UTC midnight today. Negative = net loss in R.

    R for one closed deal = deal.profit / recorded dollar_risk for
    that deal's position ticket. Deals with no matching recorded risk
    are skipped (not counted as 0 or guessed).
    """
    now = datetime.now(timezone.utc)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    deals = mt5.history_deals_get(day_start, now, group=symbol)

    if deals is None:
        error = mt5.last_error()
        if error[0] != 1:
            raise RuntimeError(f"mt5.history_deals_get() failed: {error}")
        deals = ()

    total_r = 0.0

    for deal in deals:
        if deal.entry != mt5.DEAL_ENTRY_OUT:
            continue

        dollar_risk = persistence.get_trade_risk(deal.position_id)

        if dollar_risk is None or dollar_risk <= 0:
            print(
                f"[daily-r] skipping deal on position {deal.position_id}: "
                f"no recorded initial risk (manual trade, or opened "
                f"before this tracking existed)."
            )
            continue

        total_r += deal.profit / dollar_risk

    return total_r


_PEAK_EQUITY_FILE_DEFAULT = Path("state/equity_peak.json")


def get_drawdown_pct(
    current_equity: float,
    peak_file: Path = _PEAK_EQUITY_FILE_DEFAULT,
) -> float:
    """
    Percentage drawdown from the highest equity ever observed by this
    function. Persists the peak to `peak_file` so it survives process
    restarts. Returns a POSITIVE percentage (e.g. 7.5 means 7.5%
    below peak).
    """
    peak = current_equity
    stored_peak: float | None = None

    if peak_file.exists():
        try:
            data = json.loads(peak_file.read_text(encoding="utf-8"))
            stored_peak = float(data.get("peak_equity", current_equity))
            peak = max(stored_peak, current_equity)
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            peak = current_equity
    else:
        peak_file.parent.mkdir(exist_ok=True)

    if stored_peak is None or peak != stored_peak:
        peak_file.write_text(
            json.dumps({"peak_equity": peak}),
            encoding="utf-8",
        )

    if peak <= 0:
        return 0.0

    drawdown_pct = ((peak - current_equity) / peak) * 100.0
    return max(0.0, drawdown_pct)


__all__ = ["get_daily_realized_r", "get_drawdown_pct"]