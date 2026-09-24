"""
Dry-check of phase_04_live/runtime/loop.py's setup steps (1-3),
stopping BEFORE the infinite market_source.stream() loop, so every
VERIFY-marked constructor call can be checked individually without
getting stuck inside a generator that never returns.

Safe to run against the real MT5 demo connection: nothing here
sends an order (get_symbol_trading_specs only reads via
mt5.symbol_info(), a read-only call).
"""

from datetime import datetime, timezone
from pathlib import Path

from phase_03_paper.sessions.engine import SessionEngine
from phase_03_paper.signals.adapter import StrategyAdapter

from phase_04_live.broker.connection import BrokerConnection
from phase_04_live.market.live_source import LiveMarketSource
from phase_04_live.notifications.engine import Phase4NotificationEngine
from phase_04_live.orders.order_manager import OrderManagerConfig
from phase_04_live.persistence.store import Phase4Persistence
from phase_04_live.recovery.restart import restart
from phase_04_live.risk.gates import (
    DailyLossLimitConfig,
    DrawdownLimitConfig,
    PositionLimitConfig,
)
from phase_04_live.risk.kill_switch import EmergencyKillSwitch
from phase_04_live.risk.sizing import LiveRiskConfig, get_symbol_trading_specs


SYMBOL = "XAUUSD"
TIMEFRAME = "M5"
TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
DB_PATH = "phase_04_live.db"
KILL_SWITCH_PATH = Path("state/kill_switch.json")


def check(label, fn):
    print(f"[{label}] starting...")
    try:
        result = fn()
        print(f"[{label}] OK")
        return result
    except Exception as exc:
        print(f"[{label}] FAILED: {type(exc).__name__}: {exc}")
        raise


connection = check(
    "BrokerConnection.connect",
    lambda: (
        BrokerConnection(terminal_path=TERMINAL_PATH),
    )[0],
)
connection.connect()
print(f"[BrokerConnection] is_connected={connection.is_connected()}")

persistence = check(
    "Phase4Persistence",
    lambda: Phase4Persistence(DB_PATH),
)

kill_switch = check(
    "EmergencyKillSwitch",
    lambda: EmergencyKillSwitch(KILL_SWITCH_PATH),
)

notification_engine = check(
    "Phase4NotificationEngine",
    lambda: Phase4NotificationEngine(),
)

restart_result = check(
    "restart()",
    lambda: restart(
        connection,
        persistence,
        kill_switch,
        notification_engine=notification_engine,
    ),
)
print(f"    safe_to_proceed={restart_result.safe_to_proceed}")
print(f"    kill_switch_engaged={restart_result.kill_switch_engaged}")
print(f"    review_required count={len(restart_result.review_required)}")

session_engine = check(
    "SessionEngine",
    lambda: SessionEngine(),
)

market_source = check(
    "LiveMarketSource (construction only, not stream())",
    lambda: LiveMarketSource(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        terminal_path=TERMINAL_PATH,
        poll_interval_seconds=5.0,
    ),
)

adapter = check(
    "StrategyAdapter",
    lambda: StrategyAdapter(),
)

order_config = check(
    "OrderManagerConfig + validate()",
    lambda: OrderManagerConfig(dry_run=True),
)
order_config.validate()

live_risk_config = check(
    "LiveRiskConfig",
    lambda: LiveRiskConfig(),
)

position_limit_config = check(
    "PositionLimitConfig",
    lambda: PositionLimitConfig(),
)

daily_loss_config = check(
    "DailyLossLimitConfig",
    lambda: DailyLossLimitConfig(),
)

drawdown_config = check(
    "DrawdownLimitConfig",
    lambda: DrawdownLimitConfig(),
)

specs = check(
    "get_symbol_trading_specs (real MT5 read)",
    lambda: get_symbol_trading_specs(SYMBOL),
)
print(f"    specs={specs}")

print()
print("ALL SETUP STEPS PASSED. Safe to proceed to testing the candle loop.")
