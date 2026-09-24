"""
Bounded test of the candle-processing body of loop.py's Step 4,
WITHOUT the infinite market_source.stream() loop -- pulls a fixed
number of already-closed real candles via fetch_latest() instead, so
this terminates on its own and can be watched safely.

dry_run=True throughout. No order is ever sent to the broker --
guard_and_place_order() with dry_run config only constructs and
logs the request.
"""

from datetime import datetime, timezone
from pathlib import Path

from phase_03_paper.market.engine import Candle as StrategyCandle
from phase_03_paper.sessions.engine import SessionEngine
from phase_03_paper.signals.adapter import StrategyAdapter

from phase_04_live.broker.connection import BrokerConnection
from phase_04_live.execution.idempotency import (
    build_execution_identity,
    encode_execution_identity,
    guard_and_place_order,
)
from phase_04_live.market.live_source import LiveMarketSource
from phase_04_live.orders.order_manager import OrderManagerConfig
from phase_04_live.persistence.store import Phase4Persistence
from phase_04_live.risk.gates import (
    DailyLossLimitConfig,
    DrawdownLimitConfig,
    PositionLimitConfig,
    RiskGateInput,
    evaluate_risk_gate,
)
from phase_04_live.risk.kill_switch import EmergencyKillSwitch
from phase_04_live.risk.sizing import (
    LiveRiskConfig,
    compute_position_size,
    get_account_info,
    get_symbol_trading_specs,
)


SYMBOL = "XAUUSD"
TIMEFRAME = "M5"
TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
N_CANDLES = 300  # enough recent history to plausibly find a signal


connection = BrokerConnection(terminal_path=TERMINAL_PATH)
connection.connect()
print(f"connected={connection.is_connected()}")

kill_switch = EmergencyKillSwitch(Path("state/kill_switch.json"))
persistence = Phase4Persistence("phase_04_live_test.db")  # separate test DB

session_engine = SessionEngine()

market_source = LiveMarketSource(
    symbol=SYMBOL,
    timeframe=TIMEFRAME,
    terminal_path=TERMINAL_PATH,
)
market_source.connect()

adapter = StrategyAdapter()
order_config = OrderManagerConfig(dry_run=True)
order_config.validate()

live_risk_config = LiveRiskConfig()
position_limit_config = PositionLimitConfig()
daily_loss_config = DailyLossLimitConfig()
drawdown_config = DrawdownLimitConfig()

specs = get_symbol_trading_specs(SYMBOL)

candles = market_source.fetch_latest(count=N_CANDLES)
print(f"Fetched {len(candles)} real closed candles.")
print(f"Range: {candles[0].timestamp} -> {candles[-1].timestamp}")
print()

open_positions_count = 0
open_positions_count_for_symbol = 0
total_triggered = 0
total_orders_attempted = 0

for raw_candle in candles:

    candle = StrategyCandle(
        timestamp=raw_candle.timestamp,
        open=raw_candle.open,
        high=raw_candle.high,
        low=raw_candle.low,
        close=raw_candle.close,
    )

    now = datetime.now(timezone.utc)

    session_engine.evaluate(now=now)
    current_session = session_engine.current_session(now=now)

    trading_permitted_by_session = (
        current_session is not None
        and session_engine.is_session_enabled(
            current_session.occurrence_id,
            now=now,
        )
    )

    try:
        triggered_events = adapter.on_candle(candle)
    except Exception as exc:
        print(f"[adapter] error: {exc}")
        continue

    if not triggered_events:
        continue

    total_triggered += len(triggered_events)
    print(f"[{candle.timestamp}] {len(triggered_events)} signal(s) triggered")

    if not trading_permitted_by_session:
        print("    session does not permit trading -- skipping submission")
        continue

    for event in triggered_events:

        kill_switch_engaged = kill_switch.is_engaged()
        account = get_account_info()

        sizing = compute_position_size(
            event, account, live_risk_config, specs,
        )

        if not sizing.accepted:
            print(f"    [sizing] rejected: {sizing.reason}")
            continue

        risk_input = RiskGateInput(
            symbol=SYMBOL,
            direction=event.signal.signal_type,
            open_positions_count=open_positions_count,
            open_positions_count_for_symbol=open_positions_count_for_symbol,
            kill_switch_engaged=kill_switch_engaged,
            daily_realized_r=None,
            daily_realized_amount=None,
            current_drawdown_r=None,
            current_drawdown_pct=None,
        )

        risk_result = evaluate_risk_gate(
            risk_input, position_limit_config, daily_loss_config, drawdown_config,
        )

        if not risk_result.allowed:
            print(f"    [risk] blocked: {risk_result.reason}")
            continue

        total_orders_attempted += 1

        result = guard_and_place_order(
            event, sizing, specs, order_config, store=persistence,
        )

        if result.request is not None:
            identity_key = encode_execution_identity(
                build_execution_identity(event)
            )
            persistence.record_order_attempt(
                identity_key, result.request, attempted_at=now,
            )

        print(
            f"    [order] accepted={result.accepted} "
            f"dry_run={result.dry_run} reason={result.reason}"
        )

market_source.disconnect()

print()
print("=" * 60)
print(f"Candles processed: {len(candles)}")
print(f"Signals triggered: {total_triggered}")
print(f"Order attempts (dry-run): {total_orders_attempted}")
print("=" * 60)
