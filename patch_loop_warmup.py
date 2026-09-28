path = "phase_04_live/runtime/loop.py"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

# 1. constants
anchor = "def _utc_now() -> datetime:\n"
assert src.count(anchor) == 1, "utc_now anchor"
consts = '''# Closed M5 candles replayed into StrategyAdapter at startup so swings,
# structure, zones, liquidity and the 1H+15M MTF context exist before the
# first live candle (about two weeks of gold M5 bars). Signals that trigger
# during this replay are DISCARDED -- they are history, not trades.
WARMUP_BARS = 3000

# Print a status line every N live candles (12 x M5 = 1 hour) so a quiet
# loop can be told apart from a stuck one.
HEARTBEAT_EVERY = 12

# Safety: --live is refused unless the connected MT5 account is a DEMO
# account. Change this only as a deliberate, reviewed decision.
REQUIRE_DEMO_FOR_LIVE = True


'''
src = src.replace(anchor, consts + anchor, 1)

# 2. demo-only guard for --live
anchor = "    persistence = Phase4Persistence(db_path)\n"
assert src.count(anchor) == 1, "persistence anchor"
guard = '''    if not dry_run and REQUIRE_DEMO_FOR_LIVE:
        import MetaTrader5 as _mt5

        _info = _mt5.account_info()
        if _info is None or _info.trade_mode != _mt5.ACCOUNT_TRADE_MODE_DEMO:
            raise RuntimeError(
                "Refusing to run with --live: the connected MT5 account "
                "is not confirmed to be a DEMO account."
            )

'''
src = src.replace(anchor, guard + anchor, 1)

# 3. warm-up before the candle loop
anchor = "    for raw_candle in market_source.stream():\n"
assert src.count(anchor) == 1, "stream loop anchor"
warm = '''    # ================================================================
    # STEP 3b: STRATEGY WARM-UP
    #
    # Replays recent CLOSED candles into StrategyAdapter so the strategy
    # (and its 1H+15M MTF gate) does not start cold. Events triggered
    # during the replay are discarded. Fails closed: if history cannot be
    # loaded, the loop refuses to start rather than trading blind.
    # ================================================================

    market_source.connect()

    try:
        warmup_candles = market_source.fetch_latest(WARMUP_BARS)
    except Exception as exc:
        raise RuntimeError(
            f"Warm-up history could not be loaded: {exc}"
        ) from exc

    if not warmup_candles:
        raise RuntimeError(
            "Warm-up returned no candles; refusing to start cold."
        )

    _warmup_discarded = 0
    for _wc in warmup_candles:
        try:
            _events = adapter.on_candle(
                StrategyCandle(
                    timestamp=_wc.timestamp,
                    open=_wc.open,
                    high=_wc.high,
                    low=_wc.low,
                    close=_wc.close,
                )
            )
        except Exception as exc:
            print(f"[warmup] adapter error at {_wc.timestamp}: {exc}")
            continue
        _warmup_discarded += len(_events)

    # These candles are already consumed; stream() must only yield newer ones.
    market_source._last_seen_timestamp = warmup_candles[-1].timestamp

    print(
        f"[warmup] fed {len(warmup_candles)} candles "
        f"({warmup_candles[0].timestamp} -> {warmup_candles[-1].timestamp}), "
        f"discarded {_warmup_discarded} historical signal(s), "
        f"pending={adapter.pending_count}, "
        f"adapter_errors={len(adapter.errors)}"
    )

    _candles_seen = [0]

'''
src = src.replace(anchor, warm + anchor, 1)

# 4. heartbeat inside the loop
anchor = "        if not triggered_events:\n            continue\n"
assert src.count(anchor) == 1, "triggered_events anchor"
hb = '''        _candles_seen[0] += 1
        if _candles_seen[0] % HEARTBEAT_EVERY == 0:
            print(
                f"[heartbeat] live_candles={_candles_seen[0]} "
                f"last={candle.timestamp.isoformat()} "
                f"adapter_candles={adapter.candle_count} "
                f"pending={adapter.pending_count} "
                f"adapter_errors={len(adapter.errors)}"
            )

'''
src = src.replace(anchor, hb + anchor, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(src)
print("loop.py patched; all 4 anchors matched.")
