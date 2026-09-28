"""
Normalized market event models.

TRADE EVENT
-----------
timestamp   epoch ms, UTC
trade_id    exchange-assigned id, for dedup/ordering (optional, source-dependent)
price
quantity
side        BUY  = aggressor bought (taker bought, hit the ask)
            SELL = aggressor sold   (taker sold, hit the bid)
event_type  "TRADE"

BOOK EVENT
----------
timestamp   epoch ms, UTC
bid         best bid price
ask         best ask price
bid_size    size available at best bid
ask_size    size available at best ask
event_type  "BOOK"

These are intentionally minimal (top-of-book only, not full depth). See
market_data/README.md for why -- short version: start simple, only add
multi-level depth if top-of-book imbalance turns out to carry a signal
worth the extra complexity.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Union


class EventType(str, Enum):
    TRADE = "TRADE"
    BOOK = "BOOK"


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class TradeEvent:
    timestamp: int
    price: float
    quantity: float
    side: Side
    trade_id: Optional[int] = None
    event_type: EventType = EventType.TRADE

    def to_dict(self) -> dict:
        d = asdict(self)
        d["side"] = self.side.value
        d["event_type"] = self.event_type.value
        return d


@dataclass(frozen=True)
class BookEvent:
    timestamp: int
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    event_type: EventType = EventType.BOOK

    def to_dict(self) -> dict:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d


NormalizedMarketEvent = Union[TradeEvent, BookEvent]