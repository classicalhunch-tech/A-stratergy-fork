"""
Deterministic test of the order-attempt path: forces
trading_permitted_by_session=True for the FIRST triggered signal only,
so guard_and_place_order() -> attempt_store recording actually fires
at least once. Everything else (adapter, sizing, risk gate,
idempotency, dry_run) runs for real, unmodified.
"""

from datetime import datetime, timezone
from pathlib import Path

from phase_03_paper.market.engine import Candle as StrategyCandle
from phase_03_paper.sessions.engine import SessionEngine
from phase_03_paper.signals.adapter import StrategyAdapter

from phase_04_live.broker.connection import BrokerConnection
from phase_04_live.config import DEFAULT_CONFIG as PHASE4_DEFAULT_CONFIG
from phase_04_live.execution.idempotency import guard_and_place_order
from phase_04_live.market.live_source import LiveMarketSource
from phase_04_live.notifications.engine import Phase4NotificationEngine
from phase_04_live.orders.order_manager import OrderManagerConfig
from phase_04_live.persistence.store import Phase4Persistence
from phase_04_live.recovery.restart import restart
from phase_04_live.risk.gates import (
    DailyLossLimitConfig, DrawdownLimitConfig, PositionLimitConfig,
    RiskGateInput, evaluate_risk_gate,
)
from phase_04_live.risk.kill_switch import EmergencyKillSwitch
from phase_04_live.risk.sizing import (
    LiveRiskConfig, compute_position_size, get_account_info,
    get_symbol_trading_specs,
)

SYMBOL = "XAUUSD"
TIMEFRAME = "M5"
TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
N_CANDLES = 300

connection = BrokerConnection(terminal_path=TERMINAL_PATH)
connection.connect()
print(f"connected={connection.is_connected()}")

persistence = Phase4Persistence("phase_04_live_order_path_test.db")
kill_switch = EmergencyKillSwitch(Path("state/kill_switch_order_path_test.json"))
notification_engine = Phase4NotificationEngine()

restart_result = restart(
    connection, persistence, kill_switch, notification_engine=notification_engine,
)
print(f"[restart] safe_to_proceed={restart_result.safe_to_proceed}")
print()

session_engine = SessionEngine(config=PHASE4_DEFAULT_CONFIG)

market_source = LiveMarketSource(symbol=SYMBOL, timeframe=TIMEFRAME, terminal_path=TERMINAL_PATH)
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
print()

forced_once = False
open_positions_count = 0
open_positions_count_for_symbol = 0

for raw_candle in candles:
    candle = StrategyCandle(
        timestamp=raw_candle.timestamp, open=raw_candle.open,
        high=raw_candle.high, low=raw_candle.low, close=raw_candle.close,
    )
    now = datetime.now(timezone.utc)

    session_engine.evaluate(now=now)
    current_session = session_engine.current_session(now=now)
    trading_permitted_by_session = (
        current_session is not None
        and session_engine.is_session_enabled(current_session.occurrence_id, now=now)
    )

    try:
        triggered_events = adapter.on_candle(candle)
    except Exception as exc:
        print(f"[adapter] error: {exc}")
        continue

    if not triggered_events:
        continue

    print(f"[{candle.timestamp}] {len(triggered_events)} signal(s) triggered "
          f"(session={'OPEN' if trading_permitted_by_session else 'CLOSED'})")

    # DETERMINISTIC OVERRIDE: force the session gate open for exactly
    # the first triggered signal in this run, so the order-attempt
    # path is exercised at least once. Real gating still applies to
    # every other candle.
    if not trading_permitted_by_session and not forced_once:
        print("    [TEST OVERRIDE] forcing session OPEN for this one signal "
              "to exercise the order-attempt path")
        trading_permitted_by_session = True
        forced_once = True

    if not trading_permitted_by_session:
        print("    session does not permit trading -- skipping (real gating)")
        continue

    for event in triggered_events:
        kill_switch_engaged = kill_switch.is_engaged()
        account = get_account_info()
        sizing = compute_position_size(event, account, live_risk_config, specs)
        if not sizing.accepted:
            print(f"    [sizing] rejected: {sizing.reason}")
            continue

        risk_input = RiskGateInput(
            symbol=SYMBOL, direction=event.signal.signal_type,
            open_positions_count=open_positions_count,
            open_positions_count_for_symbol=open_positions_count_for_symbol,
            has_opposite_direction_open=False,
            kill_switch_engaged=kill_switch_engaged,
            daily_realized_r=None, daily_realized_amount=None,
            current_drawdown_r=None, current_drawdown_pct=None,
        )
        risk_result = evaluate_risk_gate(
            risk_input, position_limit_config, daily_loss_config, drawdown_config,
        )
        if not risk_result.allowed:
            print(f"    [risk] blocked: {risk_result.reason}")
            continue

        pending_before = len(persistence.load_pending_order_attempts())

        result = guard_and_place_order(
            event, sizing, specs, order_config,
            store=persistence, attempt_store=persistence,
        )

        pending_after = len(persistence.load_pending_order_attempts())

        print(f"    [order] accepted={result.accepted} dry_run={result.dry_run} "
              f"reason={result.reason}")
        print(f"    [attempt_store] pending before={pending_before} after={pending_after}")

market_source.disconnect()

print()
print("=" * 60)
pending = persistence.load_pending_order_attempts()
print(f"Final pending order attempts: {len(pending)}")
for p in pending:
    print(f"    identity_key={p.identity_key} symbol={p.request.symbol} "
          f"volume={p.request.volume} attempted_at={p.attempted_at}")
print("=" * 60)
