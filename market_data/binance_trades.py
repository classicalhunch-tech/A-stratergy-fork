"""
market_data/binance_trades.py

Downloads and normalizes historical Binance spot aggTrades into TRADE events.

Source:
    https://data.binance.vision

Daily archive pattern:
    https://data.binance.vision/data/spot/daily/aggTrades/{SYMBOL}/
    {SYMBOL}-aggTrades-{YYYY-MM-DD}.zip

IMPORTANT
---------
These are Binance AGGREGATED trades, not necessarily individual executions.

The Binance aggTrade record contains:
    Aggregate tradeId
    Price
    Quantity
    First tradeId
    Last tradeId
    Timestamp
    Was the buyer the maker
    Was the trade the best price match

Aggressor-side derivation:

    is_buyer_maker == True
        -> buyer was passive/maker
        -> seller was taker/aggressor
        -> SELL

    is_buyer_maker == False
        -> buyer was taker/aggressor
        -> BUY

Therefore the resulting side is exchange-reported aggressor
classification, NOT an OHLCV-derived estimate.

CAUSALITY
---------
This module only normalizes information contained in the downloaded
historical records. It does not calculate future-looking features.

ORDERING
--------
Events are sorted by:

    timestamp
    trade_id

This gives deterministic ordering when multiple trades share the
same millisecond timestamp.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date, timedelta
from typing import Iterator, List

import pandas as pd
import numpy as np
import requests

from .models import Side, TradeEvent


BASE_URL = (
    "https://data.binance.vision/data/spot/daily/aggTrades/"
    "{symbol}/{symbol}-aggTrades-{day}.zip"
)


COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "timestamp",
    "is_buyer_maker",
    "is_best_match",
]


def _daterange(start: date, end: date) -> Iterator[date]:
    """Yield every date from start through end, inclusive."""
    if end < start:
        raise ValueError("end date must be >= start date")

    current = start

    while current <= end:
        yield current
        current += timedelta(days=1)


def _download_one_day(
    symbol: str,
    day: date,
    session: requests.Session,
) -> pd.DataFrame:
    """
    Download and parse one Binance spot aggTrades archive.

    Returns an empty DataFrame when Binance has no archive for the day.
    """
    symbol = symbol.upper().strip()

    if not symbol:
        raise ValueError("symbol must not be empty")

    url = BASE_URL.format(
        symbol=symbol,
        day=day.isoformat(),
    )

    response = session.get(url, timeout=30)

    if response.status_code == 404:
        print(f"  [skip] {day}: archive not found")
        return pd.DataFrame(columns=COLUMNS)

    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        members = [
            name
            for name in zf.namelist()
            if not name.endswith("/")
        ]

        if not members:
            raise ValueError(f"No data file found inside archive: {url}")

        inner_name = members[0]

        with zf.open(inner_name) as f:
            raw_head = f.read(500)

        first_field = (
            raw_head
            .split(b",", 1)[0]
            .decode("utf-8", errors="ignore")
            .strip()
        )

        # Binance archives may contain a header row.
        # Data rows begin with a numeric aggregate trade ID.
        has_header = not first_field.lstrip("-").isdigit()

        with zf.open(inner_name) as f:
            df = pd.read_csv(
                f,
                header=0 if has_header else None,
                names=COLUMNS,
                dtype={
                    "agg_trade_id": "int64",
                    "price": "float64",
                    "quantity": "float64",
                    "first_trade_id": "int64",
                    "last_trade_id": "int64",
                    "timestamp": "int64",
                    "is_buyer_maker": "bool",
                    "is_best_match": "bool",
                },
            )

    df = _normalize_timestamp_unit(df, symbol, day)

    _validate_raw_trades(df, symbol, day)

    print(f"  [ok]   {day}: {len(df):,} aggregate trades")

    return df


def _normalize_timestamp_unit(
    df: pd.DataFrame,
    symbol: str,
    day: date,
) -> pd.DataFrame:
    """
    Detect whether Binance's 'timestamp' column is epoch milliseconds or
    epoch microseconds, and convert to a canonical unit: epoch milliseconds.

    Binance switched some daily archives from ms to us precision starting
    mid-2025. Silently assuming ms would corrupt every downstream causal
    time-window calculation (timestamps would parse to a date far in the
    future, off by a factor of 1000). Detection is by magnitude:

        epoch ms for 2020-2030   is  ~13 digits (order 1e12-1e13)
        epoch us for 2020-2030   is  ~16 digits (order 1e15-1e16)
    """
    if df.empty:
        return df

    sample = int(df["timestamp"].iloc[0])

    if sample >= 1_000_000_000_000_000:  # microseconds
        print(f"  [note] {day}: detected microsecond timestamps -> converting to ms")
        df = df.copy()
        df["timestamp"] = (df["timestamp"] // 1000).astype("int64")
    elif sample >= 1_000_000_000_000:  # already milliseconds
        pass
    else:
        raise ValueError(
            f"{symbol} {day}: timestamp magnitude {sample} doesn't look like "
            "epoch ms or epoch us -- refusing to guess"
        )

    return df


def _validate_raw_trades(
    df: pd.DataFrame,
    symbol: str,
    day: date,
) -> None:
    """Validate the minimum invariants required by the normalizer."""

    missing = [column for column in COLUMNS if column not in df.columns]

    if missing:
        raise ValueError(
            f"{symbol} {day}: missing Binance columns: {missing}"
        )

    if df.empty:
        return

    if (df["price"] <= 0).any():
        raise ValueError(
            f"{symbol} {day}: found non-positive trade price"
        )

    if (df["quantity"] <= 0).any():
        raise ValueError(
            f"{symbol} {day}: found non-positive trade quantity"
        )

    if (df["agg_trade_id"].duplicated()).any():
        raise ValueError(
            f"{symbol} {day}: duplicate aggregate trade IDs detected"
        )

    invalid_side_values = set(
        df["is_buyer_maker"].dropna().unique()
    ) - {True, False}

    if invalid_side_values:
        raise ValueError(
            f"{symbol} {day}: unexpected is_buyer_maker values: "
            f"{invalid_side_values}"
        )


def download_agg_trades(
    symbol: str,
    start_day: date,
    end_day: date,
) -> pd.DataFrame:
    """
    Download and concatenate daily Binance spot aggTrades.

    Date range is inclusive.
    """
    symbol = symbol.upper().strip()

    if not symbol:
        raise ValueError("symbol must not be empty")

    session = requests.Session()

    frames: List[pd.DataFrame] = []

    try:
        for day in _daterange(start_day, end_day):
            frames.append(
                _download_one_day(
                    symbol=symbol,
                    day=day,
                    session=session,
                )
            )
    finally:
        session.close()

    if not frames:
        return pd.DataFrame(columns=COLUMNS)

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    if result.empty:
        return pd.DataFrame(columns=COLUMNS)

    # Deterministic global ordering.
    result = (
        result
        .sort_values(
            ["timestamp", "agg_trade_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    # Aggregate trade IDs should be unique across the downloaded range.
    if result["agg_trade_id"].duplicated().any():
        duplicates = int(result["agg_trade_id"].duplicated().sum())

        raise ValueError(
            f"{symbol}: {duplicates} duplicate aggregate trade IDs "
            "found across downloaded files"
        )

    return result


def normalize_trades(
    raw: pd.DataFrame,
) -> List[TradeEvent]:
    """
    Convert Binance aggTrade rows into normalized TradeEvent objects.

    The resulting side is based directly on Binance's
    is_buyer_maker field.

    NOTE: this creates one Python object per row. Fine for a day or two of
    data (hundreds of thousands of rows); for bulk multi-week/multi-month
    processing, use normalize_trades_df() instead, which stays in pandas
    and avoids per-row object overhead.
    """
    if raw.empty:
        return []

    missing = [
        column
        for column in COLUMNS
        if column not in raw.columns
    ]

    if missing:
        raise ValueError(
            f"Cannot normalize trades; missing columns: {missing}"
        )

    events: List[TradeEvent] = []

    for row in raw.itertuples(index=False):
        # Binance semantics:
        #
        # buyer is maker
        #     -> seller is taker/aggressor
        #     -> SELL
        #
        # buyer is taker
        #     -> buyer is aggressor
        #     -> BUY
        side = (
            Side.SELL
            if bool(row.is_buyer_maker)
            else Side.BUY
        )

        events.append(
            TradeEvent(
                timestamp=int(row.timestamp),
                price=float(row.price),
                quantity=float(row.quantity),
                side=side,
                trade_id=int(row.agg_trade_id),
            )
        )

    return events


def trades_to_dataframe(
    events: List[TradeEvent],
) -> pd.DataFrame:
    """
    Convert normalized TradeEvent objects into a research DataFrame.

    Ordering is deterministic:
        timestamp -> trade_id
    """
    if not events:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "price",
                "quantity",
                "side",
                "trade_id",
                "event_type",
            ]
        )

    df = pd.DataFrame(
        [event.to_dict() for event in events]
    )

    df = (
        df
        .sort_values(
            ["timestamp", "trade_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return df


def normalize_trades_df(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized equivalent of normalize_trades() + trades_to_dataframe(),
    without creating one Python object per row.

    Use this for bulk processing (multi-day / multi-week downloads) where
    normalize_trades()'s per-row object creation would be slow and memory
    heavy at millions of rows. Produces the same schema and same
    timestamp -> trade_id ordering; same aggressor-side semantics
    (is_buyer_maker True -> SELL, False -> BUY).
    """
    if raw.empty:
        return pd.DataFrame(columns=["timestamp", "price", "quantity", "side", "trade_id", "event_type"])

    missing = [column for column in COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Cannot normalize trades; missing columns: {missing}")

    df = pd.DataFrame(
        {
            "timestamp": raw["timestamp"].astype("int64"),
            "price": raw["price"].astype("float64"),
            "quantity": raw["quantity"].astype("float64"),
            "side": np.where(raw["is_buyer_maker"], "SELL", "BUY"),
            "trade_id": raw["agg_trade_id"].astype("int64"),
        }
    )
    df["event_type"] = "TRADE"

    df = df.sort_values(["timestamp", "trade_id"], kind="mergesort").reset_index(drop=True)
    return df