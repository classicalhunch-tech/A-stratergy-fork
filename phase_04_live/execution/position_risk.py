"""
phase_04_live/execution/position_risk.py

Bridge between an accepted live order and the trade_risk record.

    order ticket -> deal ticket -> deal.position_id
                 -> persistence.record_trade_risk(position_id, dollar_risk)

The broker layer (broker/mt5.py) owns the MT5 resolution. This module
only decides WHEN to ask and what to do with the answer.

If the position ID cannot be confirmed, NOTHING is recorded and the
order ticket is never used as a stand-in. The execution is left to the
durable order-attempt / recovery mechanism.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from phase_04_live.broker.mt5 import resolve_position_id

logger = logging.getLogger(__name__)


def record_risk_for_accepted_order(
    connection,
    result,
    dollar_risk: float,
    persistence,
    *,
    attempts: int = 5,
    delay_seconds: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
    resolver: Callable[..., Optional[int]] = resolve_position_id,
) -> Optional[int]:
    """
    Record dollar_risk against the ACTUAL broker position ID.

    Returns the position ID when it was confirmed and recorded,
    otherwise None (nothing recorded).

    Only real, accepted orders are processed. Dry-run, rejected and
    ticket-less results are ignored.
    """

    if not result.accepted or result.dry_run or result.ticket is None:
        return None

    deal_ticket = getattr(result, "deal", None)

    for attempt in range(attempts):
        position_id = resolver(
            connection,
            result.ticket,
            deal_ticket=deal_ticket,
        )

        if position_id is not None:
            persistence.record_trade_risk(position_id, dollar_risk)
            return position_id

        if attempt < attempts - 1:
            sleep(delay_seconds)

    logger.warning(
        "Position ID for order ticket %s not confirmed after %d "
        "attempts; trade risk NOT recorded. Leaving to order-attempt "
        "recovery.",
        result.ticket,
        attempts,
    )
    return None
