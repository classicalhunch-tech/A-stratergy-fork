
"""
market_data.py

Fetches multi-timeframe OHLC candle data for Gold.

Research data source:
    yfinance -> GC=F (COMEX Gold Futures)

IMPORTANT:
GC=F is a RESEARCH PROXY for XAUUSD spot. It is suitable for early
strategy development and backtesting, but it is NOT the same as the
broker's XAUUSD feed.

Before demo/live trading, this module should be switched to the
broker's actual XAUUSD data source, such as MT5.

The rest of the bot should continue receiving the same standardized
DataFrame structure:
    Open, High, Low, Close, Volume

4H candles:
    4H candles are constructed from 1H candles using UTC boundaries:
    00:00, 04:00, 08:00, 12:00, 16:00, 20:00.

A 4H candle is considered complete only when all four underlying
1H candles are available.
"""

import logging
import time

import pandas as pd
import requests
import yfinance as yf


logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


TIMEFRAME_MAP = {
    "5M": "5m",
    "15M": "15m",
    "1H": "1h",
    "4H": None,
    "D": "1d",
}


MAX_PERIOD = {
    "5m": "60d",
    "15m": "60d",
    "1h": "730d",
    "1d": "10y",
}


DEFAULT_TICKER = "GC=F"


MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2


RETRYABLE_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.HTTPError,
    ConnectionError,
    TimeoutError,
)


class MarketDataError(Exception):
    """Raised when market data cannot be fetched after all retries."""
    pass


class MarketDataValidationError(Exception):
    """Raised when candle data fails a critical validation check."""
    pass


def _resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Build 4H candles from complete groups of four 1H candles.

    A 4H candle is only created when all four underlying 1H candles
    are present. This prevents incomplete 4H candles from being used.
    """
    if df_1h.empty:
        return df_1h

    df_1h = df_1h.sort_index()

    resampled = df_1h.resample("4h").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )

    candle_count = df_1h["Close"].resample("4h").count()

    resampled = resampled[candle_count == 4]

    resampled = resampled.dropna(subset=["Open", "High", "Low", "Close"])

    return resampled


def remove_incomplete_candle(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Remove the currently forming candle.

    The strategy should normally operate only on completed candles.
    """
    if df.empty:
        return df

    timeframe_minutes = {
        "5M": 5,
        "15M": 15,
        "1H": 60,
        "4H": 240,
        "D": 1440,
    }

    timeframe = timeframe.upper()

    if timeframe not in timeframe_minutes:
        raise ValueError(f"Unknown timeframe '{timeframe}'.")

    candle_duration = pd.Timedelta(minutes=timeframe_minutes[timeframe])

    now = pd.Timestamp.now(tz="UTC")

    last_timestamp = df.index[-1]

    if last_timestamp + candle_duration > now:
        return df.iloc[:-1].copy()

    return df


def _fetch_with_retry(ticker: str, period: str, interval: str) -> pd.DataFrame:

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            raw = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
            )

            if raw.empty:
                logger.warning(
                    "yfinance returned no data for %s @ %s (attempt %d/%d)",
                    ticker, interval, attempt, MAX_RETRIES,
                )
                return raw

            return raw

        except RETRYABLE_EXCEPTIONS as exc:
            last_error = exc

            logger.warning(
                "Network error fetching %s @ %s (attempt %d/%d): %s",
                ticker, interval, attempt, MAX_RETRIES, exc,
            )

            if attempt < MAX_RETRIES:
                sleep_time = RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
                logger.info("Retrying in %ds...", sleep_time)
                time.sleep(sleep_time)

    raise MarketDataError(
        f"Failed to fetch {ticker} @ {interval} after {MAX_RETRIES} attempts: {last_error}"
    )


def validate_candles(df: pd.DataFrame, timeframe: str = "") -> dict:
    """
    Validate OHLC candle data.

    Returns:
        {"warnings": [...], "critical": [...]}
    """
    warnings = []
    critical = []

    if df.empty:
        critical.append("No candle data returned.")
        return {"warnings": warnings, "critical": critical}

    label = f" ({timeframe})" if timeframe else ""

    if not df.index.is_monotonic_increasing:
        critical.append(f"Timestamps are not sorted ascending{label}.")

    duplicate_count = df.index.duplicated().sum()
    if duplicate_count > 0:
        critical.append(f"{duplicate_count} duplicate timestamp(s) found{label}.")

    required_columns = ["Open", "High", "Low", "Close"]

    missing_columns = [c for c in required_columns if c not in df.columns]

    if missing_columns:
        critical.append(f"Missing required column(s): {missing_columns}{label}.")
        return {"warnings": warnings, "critical": critical}

    nan_count = df[required_columns].isna().sum().sum()
    if nan_count > 0:
        critical.append(f"{nan_count} missing OHLC value(s) found{label}.")

    clean = df.dropna(subset=required_columns)

    bad_high = clean[clean["High"] < clean[["Open", "Close", "Low"]].max(axis=1)]
    if not bad_high.empty:
        critical.append(f"{len(bad_high)} candle(s) where High is not the maximum{label}.")

    bad_low = clean[clean["Low"] > clean[["Open", "Close", "High"]].min(axis=1)]
    if not bad_low.empty:
        critical.append(f"{len(bad_low)} candle(s) where Low is not the minimum{label}.")

    non_positive = clean[(clean[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)]
    if not non_positive.empty:
        critical.append(f"{len(non_positive)} candle(s) with zero/negative price{label}.")

    if len(clean) > 1:
        deltas = clean.index.to_series().diff().dropna()
        median_delta = deltas.median()
        large_gaps = deltas[deltas > median_delta * 5]
        if len(large_gaps) > 0:
            warnings.append(
                f"{len(large_gaps)} unusually large time gap(s) found{label} "
                f"(median interval: {median_delta})."
            )

    return {"warnings": warnings, "critical": critical}


def get_candles(
    timeframe: str,
    ticker: str = DEFAULT_TICKER,
    period: str = None,
    validate: bool = True,
) -> pd.DataFrame:

    timeframe = timeframe.upper()

    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(
            f"Unknown timeframe '{timeframe}'. Valid options: {list(TIMEFRAME_MAP.keys())}"
        )

    if timeframe == "4H":
        df_1h = get_candles("1H", ticker=ticker, period=period, validate=False)

        result = _resample_to_4h(df_1h)

        result = remove_incomplete_candle(result, timeframe)

        if validate:
            validation = validate_candles(result, timeframe="4H")

            for issue in validation["warnings"]:
                logger.warning(issue)

            for issue in validation["critical"]:
                logger.error(issue)

            if validation["critical"]:
                raise MarketDataValidationError(
                    "Critical validation failure for 4H: " + "; ".join(validation["critical"])
                )

        return result

    interval = TIMEFRAME_MAP[timeframe]

    fetch_period = period or MAX_PERIOD[interval]

    raw = _fetch_with_retry(ticker, fetch_period, interval)

    if raw.empty:
        return raw

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC")
    else:
        raw.index = raw.index.tz_convert("UTC")

    result = raw[["Open", "High", "Low", "Close", "Volume"]].copy()

    result = remove_incomplete_candle(result, timeframe)

    if validate:
        validation = validate_candles(result, timeframe=timeframe)

        for issue in validation["warnings"]:
            logger.warning(issue)

        for issue in validation["critical"]:
            logger.error(issue)

        if validation["critical"]:
            raise MarketDataValidationError(
                f"Critical validation failure for {timeframe}: "
                + "; ".join(validation["critical"])
            )

    return result


def get_all_timeframes(ticker: str = DEFAULT_TICKER) -> dict:

    results = {}

    for tf in ["D", "4H", "1H", "15M", "5M"]:
        try:
            results[tf] = get_candles(tf, ticker=ticker)
        except (MarketDataError, MarketDataValidationError) as exc:
            logger.error("Could not fetch %s data: %s", tf, exc)
            results[tf] = pd.DataFrame()

    return results


if __name__ == "__main__":

    for tf in ["D", "4H", "1H", "15M", "5M"]:
        try:
            df = get_candles(tf)

            print(f"\n--- {tf} ({len(df)} candles) ---")

            if not df.empty:
                print(df.tail(3))

                validation = validate_candles(df, timeframe=tf)

                if validation["warnings"]:
                    print("Warnings:", validation["warnings"])

                if validation["critical"]:
                    print("Critical:", validation["critical"])

                if not validation["warnings"] and not validation["critical"]:
                    print("Validation: clean")
            else:
                print("No data returned.")

        except (MarketDataError, MarketDataValidationError) as exc:
            print(f"\n--- {tf}: FAILED ({exc}) ---")