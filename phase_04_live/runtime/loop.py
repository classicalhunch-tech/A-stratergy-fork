"""
phase_04_live/runtime/loop.py

Phase 4 live runtime daemon loop.

Ties together:

    broker connection
        -> restart / recovery
        -> session gating
        -> live market data
        -> existing strategy pipeline via StrategyAdapter
        -> live broker position state
        -> pre-trade risk gate
        -> position sizing
        -> idempotency-guarded order submission
        -> notifications

ARCHITECTURE RULES
------------------

1. Phase 4 orchestrates the existing strategy. It does not duplicate
   strategy logic.

2. Every candle reaches StrategyAdapter, regardless of session
   permission. Session permission controls trading, not structural
   continuity.

3. Restart/recovery must report whether the runtime is safe to trade.
   If restart is not safe, the loop still consumes candles so strategy
   state remains continuous, but the kill switch blocks new trades.

4. The live broker position state is queried from MT5 for every
   triggered signal. The loop never assumes that an in-memory position
   count is authoritative.

5. The risk gate runs BEFORE position sizing.

6. Daily realized loss is measured in R.

7. The daily loss circuit breaker is set to -2R. Once reached, new
   trades are blocked for the rest of the day.

8. Drawdown is measured from persisted peak equity.

9. The drawdown circuit breaker is set to 10%. At 10% drawdown, new
   trading is blocked until the kill switch / operator state is
   manually reset according to the existing recovery architecture.

10. Position limits remain active:
        - maximum 1 open position overall
        - maximum 1 open position per symbol
        - no opposite-direction hedge

11. Order submission remains behind guard_and_place_order(), which
    provides the existing idempotency and durable order-attempt
    protection.

12. dry_run is controlled explicitly by the caller. This loop never
    silently changes it.

IMPORTANT
---------

OrderResult.ticket represents MT5 result.order, i.e. the broker ORDER
ticket returned by order_send(). OrderResult.deal is the MT5 deal ticket.

The realized-R persistence path uses closed-deal position_id, so the
order ticket is never used as the trade_risk key. After an accepted
live order, record_risk_for_accepted_order() resolves the actual
position ID from MT5 deal history (order -> deal -> position_id) and
records the dollar risk against it. If the position ID cannot be
confirmed, nothing is recorded and the durable order-attempt /
recovery path handles the execution.

OPERATOR CHOICE
---------------

_auto_trade_all_sessions() intentionally records TRADE for every
undecided session occurrence.

This overrides SessionEngine's normal undecided-session behavior and
was deliberately chosen so that live trading does not wait for a
manual per-session decision.

This loop does NOT:

    - duplicate strategy logic
    - guess broker capabilities
    - resend an order whose broker outcome is unknown
    - bypass restart/recovery
    - bypass the kill switch
    - bypass the risk gate
    - bypass idempotency
    - automatically enable live trading
    - treat an unknown broker position state as zero positions
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from phase_03_paper.market.engine import Candle as StrategyCandle
from phase_03_paper.models import SessionDecision, SessionStatus
from phase_03_paper.sessions.engine import SessionEngine
from phase_03_paper.signals.adapter import StrategyAdapter

from phase_04_live.broker.connection import BrokerConnection
from phase_04_live.broker.positions import get_position_state
from phase_04_live.config import DEFAULT_CONFIG as PHASE4_DEFAULT_CONFIG
from phase_04_live.execution.idempotency import (
    build_execution_identity,
    encode_execution_identity,
    guard_and_place_order,
)
from phase_04_live.execution.position_risk import record_risk_for_accepted_order
from phase_04_live.market.live_source import LiveMarketSource
from phase_04_live.notifications.engine import Phase4NotificationEngine
from phase_04_live.orders.order_manager import OrderManagerConfig
from phase_04_live.persistence.store import Phase4Persistence
from phase_04_live.recovery.restart import restart
from phase_04_live.risk.config_file import apply_config_file
from phase_04_live.risk.gates import (
    DailyLossLimitConfig,
    DrawdownLimitConfig,
    PositionLimitConfig,
    RiskGateInput,
    evaluate_risk_gate,
)
from phase_04_live.risk.kill_switch import EmergencyKillSwitch
from phase_04_live.risk.live_metrics import (
    get_daily_realized_r,
    get_drawdown_pct,
)
from phase_04_live.risk.sizing import (
    LiveRiskConfig,
    compute_position_size,
    get_account_info,
    get_symbol_trading_specs,
)


# Closed M5 candles replayed into StrategyAdapter at startup so swings,
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


def _utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def _auto_trade_all_sessions(
    session_engine: SessionEngine,
    now: datetime,
) -> None:
    """
    Automatically record TRADE for every undecided session occurrence.

    This is an explicit operator choice for the live runtime. It means
    the runtime does not wait for a manual YES/NO decision for each
    session occurrence.
    """

    for state in session_engine.all_states(now=now):
        if (
            state.decision == SessionDecision.UNDECIDED
            and state.status
            not in (
                SessionStatus.CLOSED,
                SessionStatus.SKIPPED,
            )
        ):
            try:
                session_engine.record_decision(
                    state.session_name,
                    SessionDecision.TRADE,
                    now=now,
                )
            except ValueError:
                # The occurrence may have become terminal between the
                # all_states() call and record_decision().
                continue


def run_live_loop(
    *,
    symbol: str,
    timeframe: str,
    terminal_path: str,
    db_path: str,
    kill_switch_path: Path,
    dry_run: bool = True,
    poll_interval_seconds: float = 5.0,
    risk_config_path: Optional[Path] = None,
) -> None:
    """
    Run the Phase 4 live trading daemon loop.

    dry_run=True:
        Build and validate orders without sending them to MT5.

    dry_run=False:
        Allow the existing order-management pipeline to submit real
        broker orders.

    risk_config_path:
        Optional JSON file written by the dashboard. When supplied,
        sizing configuration is refreshed from this file on each
        candle and again immediately before sizing.
    """

    # ================================================================
    # STEP 1: BROKER CONNECTION
    # ================================================================

    connection = BrokerConnection(
        terminal_path=terminal_path,
    )

    connection.connect()

    if not connection.is_connected():
        raise RuntimeError(
            "BrokerConnection.connect() returned without raising, "
            "but is_connected() is False. Refusing to start the "
            "runtime against an unconfirmed broker connection."
        )

    if not dry_run and REQUIRE_DEMO_FOR_LIVE:
        import MetaTrader5 as _mt5

        _info = _mt5.account_info()
        if _info is None or _info.trade_mode != _mt5.ACCOUNT_TRADE_MODE_DEMO:
            raise RuntimeError(
                "Refusing to run with --live: the connected MT5 account "
                "is not confirmed to be a DEMO account."
            )

    persistence = Phase4Persistence(db_path)

    kill_switch = EmergencyKillSwitch(
        kill_switch_path,
    )

    notification_engine = Phase4NotificationEngine()

    # ================================================================
    # STEP 2: RESTART / RECOVERY
    # ================================================================

    restart_result = restart(
        connection,
        persistence,
        kill_switch,
        notification_engine=notification_engine,
    )

    print(
        f"[restart] safe_to_proceed="
        f"{restart_result.safe_to_proceed}"
    )

    print(
        "[restart] "
        f"kill_switch_engaged_by_this_restart="
        f"{restart_result.kill_switch_engaged}"
    )

    if restart_result.review_required:
        print(
            f"[restart] {len(restart_result.review_required)} "
            "decision(s) require manual review -- see notifications."
        )

        for decision in restart_result.review_required:
            print(
                f"    - {decision.reason}"
            )

    if not restart_result.safe_to_proceed:
        print(
            "[restart] safe_to_proceed is False. "
            "Candles will continue flowing for structural continuity, "
            "but the risk gate will block new trades until the existing "
            "safety state permits trading."
        )

    # ================================================================
    # STEP 3: SESSION / MARKET / STRATEGY / ORDER CONFIGURATION
    # ================================================================

    session_engine = SessionEngine(
        config=PHASE4_DEFAULT_CONFIG,
    )

    market_source = LiveMarketSource(
        symbol=symbol,
        timeframe=timeframe,
        terminal_path=terminal_path,
        poll_interval_seconds=poll_interval_seconds,
        on_quality_issue=lambda issue: print(
            f"[quality] {issue}"
        ),
    )

    adapter = StrategyAdapter()

    order_config = OrderManagerConfig(
        dry_run=dry_run,
    )

    order_config.validate()

    live_risk_config = LiveRiskConfig()

    # ================================================================
    # PRE-TRADE RISK LIMITS
    #
    # Position protection:
    #     maximum 1 open position overall
    #     maximum 1 open position for the symbol
    #     no opposite-direction hedge
    #
    # Daily loss protection:
    #     stop opening new trades at -2R for the rest of the day
    #
    # Drawdown protection:
    #     stop trading at 10% drawdown from persisted peak equity
    # ================================================================

    position_limit_config = PositionLimitConfig()

    daily_loss_config = DailyLossLimitConfig(
        max_daily_loss_r=2.0,
    )

    drawdown_config = DrawdownLimitConfig(
        max_drawdown_pct=10.0,
    )

    specs = get_symbol_trading_specs(symbol)

    print(
        f"[startup] symbol={symbol} "
        f"timeframe={timeframe} "
        f"dry_run={dry_run}"
    )

    print(
        "[startup] risk limits: "
        "max_positions=1, "
        "max_positions_per_symbol=1, "
        "hedging=disabled, "
        "daily_loss_stop=-2R, "
        "drawdown_stop=10%"
    )

    print(
        "[startup] entering candle loop..."
    )

    # Tracks the last fixed lot size printed to the log.
    _last_logged_lot_size = [
        live_risk_config.fixed_lot_size
    ]

    # ================================================================
    # STEP 4: CONTINUOUS CANDLE LOOP
    # ================================================================

    # ================================================================
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

    # CHANGED: these candles are already consumed; stream() must only yield
    # newer ones, and the quality filter must continue from the end of this
    # history.
    market_source.mark_history_loaded(warmup_candles[-1])

    print(
        f"[warmup] fed {len(warmup_candles)} candles "
        f"({warmup_candles[0].timestamp} -> {warmup_candles[-1].timestamp}), "
        f"discarded {_warmup_discarded} historical signal(s), "
        f"pending={adapter.pending_count}, "
        f"adapter_errors={len(adapter.errors)}"
    )

    _candles_seen = [0]

    for raw_candle in market_source.stream():

        # ------------------------------------------------------------
        # Convert live market data into the existing Phase 3 Candle
        # model used by StrategyAdapter.
        # ------------------------------------------------------------

        candle = StrategyCandle(
            timestamp=raw_candle["timestamp"],
            open=raw_candle["open"],
            high=raw_candle["high"],
            low=raw_candle["low"],
            close=raw_candle["close"],
        )

        now = _utc_now()

        # ------------------------------------------------------------
        # SESSION GATING
        #
        # Session permission controls trading.
        #
        # It does NOT control whether StrategyAdapter receives the
        # candle. Structural continuity must be preserved.
        # ------------------------------------------------------------

        _auto_trade_all_sessions(
            session_engine,
            now=now,
        )

        current_session = session_engine.current_session(
            now=now,
        )

        trading_permitted_by_session = (
            current_session is not None
            and session_engine.is_session_enabled(
                current_session.occurrence_id,
                now=now,
            )
        )

        # ------------------------------------------------------------
        # REFRESH DASHBOARD-CONTROLLED SIZING CONFIGURATION
        #
        # This happens every candle so runtime configuration changes
        # are picked up without restarting the process.
        # ------------------------------------------------------------

        if risk_config_path is not None:

            diagnostic_error = apply_config_file(
                live_risk_config,
                risk_config_path,
            )

            if diagnostic_error is not None:
                print(
                    f"[risk-config] {diagnostic_error}"
                )

            elif (
                live_risk_config.fixed_lot_size
                != _last_logged_lot_size[0]
            ):
                print(
                    "[risk-config] "
                    f"fixed_lot_size changed to "
                    f"{live_risk_config.fixed_lot_size}"
                )

                _last_logged_lot_size[0] = (
                    live_risk_config.fixed_lot_size
                )

        # ------------------------------------------------------------
        # STRATEGY PIPELINE
        #
        # Every candle reaches StrategyAdapter.
        # ------------------------------------------------------------

        try:
            triggered_events = adapter.on_candle(
                candle,
            )

        except Exception as exc:
            print(
                f"[adapter] error processing candle: {exc}"
            )
            continue

        _candles_seen[0] += 1
        if _candles_seen[0] % HEARTBEAT_EVERY == 0:
            print(
                f"[heartbeat] live_candles={_candles_seen[0]} "
                f"last={candle.timestamp.isoformat()} "
                f"adapter_candles={adapter.candle_count} "
                f"pending={adapter.pending_count} "
                f"adapter_errors={len(adapter.errors)}"
            )

        if not triggered_events:
            continue

        # ------------------------------------------------------------
        # SESSION PERMISSION CHECK
        #
        # The strategy has already seen the candle and generated its
        # events. We only suppress order submission here.
        # ------------------------------------------------------------

        if not trading_permitted_by_session:
            print(
                f"[session] {len(triggered_events)} signal(s) "
                "triggered but the current session does not permit "
                "trading -- skipping order submission for this candle."
            )
            continue

        # ============================================================
        # PROCESS EACH TRIGGERED SIGNAL
        # ============================================================

        for event in triggered_events:

            # --------------------------------------------------------
            # 1. CURRENT KILL-SWITCH STATE
            # --------------------------------------------------------

            kill_switch_engaged = (
                kill_switch.is_engaged()
            )

            # --------------------------------------------------------
            # 2. CURRENT ACCOUNT STATE
            #
            # Account information is needed for drawdown and sizing.
            # --------------------------------------------------------

            try:
                account = get_account_info()

            except Exception as exc:
                print(
                    f"[account] could not read account state, "
                    f"skipping signal: {exc}"
                )
                continue

            # --------------------------------------------------------
            # 3. CURRENT BROKER POSITION STATE
            #
            # This is queried fresh for every triggered signal.
            #
            # Fail closed: if MT5 position state is unknown, we do not
            # assume there are zero positions.
            # --------------------------------------------------------

            try:
                position_state = get_position_state(
                    symbol=symbol,
                    signal_direction=event.signal.signal_type,
                )

            except RuntimeError as exc:
                print(
                    "[position] could not read broker position "
                    f"state, skipping this signal: {exc}"
                )
                continue

            # --------------------------------------------------------
            # 4. CURRENT DAILY REALIZED R
            #
            # This is the measurement used by the -2R daily-loss gate.
            # --------------------------------------------------------

            try:
                daily_realized_r = get_daily_realized_r(
                    persistence,
                    symbol,
                )

            except Exception as exc:
                # The daily-loss circuit breaker is enabled.
                # An unavailable measurement must therefore fail
                # closed rather than being treated as zero.
                print(
                    "[risk] could not calculate daily realized R, "
                    f"skipping signal for safety: {exc}"
                )
                continue

            # --------------------------------------------------------
            # 5. CURRENT EQUITY DRAWDOWN
            #
            # Drawdown is calculated against the persisted peak equity.
            # --------------------------------------------------------

            try:
                current_drawdown_pct = get_drawdown_pct(
                    account.equity,
                )

            except Exception as exc:
                # Drawdown protection is enabled.
                # An unavailable measurement must fail closed.
                print(
                    "[risk] could not calculate current drawdown, "
                    f"skipping signal for safety: {exc}"
                )
                continue

            # --------------------------------------------------------
            # 6. PRE-TRADE RISK GATE
            #
            # IMPORTANT:
            #
            # Risk is evaluated BEFORE position sizing.
            #
            # There is no reason to calculate a trade size when the
            # account is already prohibited from opening a new trade.
            # --------------------------------------------------------

            risk_input = RiskGateInput(
                symbol=symbol,
                direction=event.signal.signal_type,
                open_positions_count=(
                    position_state.open_positions_count
                ),
                open_positions_count_for_symbol=(
                    position_state.open_positions_count_for_symbol
                ),
                has_opposite_direction_open=(
                    position_state.has_opposite_direction_open
                ),
                daily_realized_r=daily_realized_r,
                daily_realized_amount=None,
                current_drawdown_r=None,
                current_drawdown_pct=current_drawdown_pct,
                kill_switch_engaged=kill_switch_engaged,
            )

            risk_result = evaluate_risk_gate(
                risk_input,
                position_limit_config,
                daily_loss_config,
                drawdown_config,
            )

            if not risk_result.allowed:
                print(
                    f"[risk] blocked: {risk_result.reason}"
                )

                # Print additional blocking reasons when more than one
                # independent risk condition is simultaneously active.
                for issue in risk_result.blocking_issues:
                    if issue != risk_result.reason:
                        print(
                            f"[risk] also blocked: {issue}"
                        )

                continue

            # --------------------------------------------------------
            # 7. REFRESH DASHBOARD-CONTROLLED SIZING CONFIGURATION
            #
            # The latest configuration is applied immediately before
            # sizing so a dashboard change is not unnecessarily stale.
            # --------------------------------------------------------

            if risk_config_path is not None:

                config_error = apply_config_file(
                    live_risk_config,
                    risk_config_path,
                )

                if config_error is not None:
                    print(
                        f"[risk-config] {config_error}"
                    )

            # --------------------------------------------------------
            # 8. POSITION SIZING
            #
            # Reached only after the pre-trade risk gate has allowed
            # the signal.
            # --------------------------------------------------------

            try:
                sizing = compute_position_size(
                    event,
                    account,
                    live_risk_config,
                    specs,
                )

            except Exception as exc:
                print(
                    f"[sizing] error calculating position size: {exc}"
                )
                continue

            if not sizing.accepted:
                print(
                    f"[sizing] rejected: {sizing.reason}"
                )
                continue

            # --------------------------------------------------------
            # 9. IDEMPOTENCY-GUARDED ORDER SUBMISSION
            #
            # guard_and_place_order() remains the single order boundary.
            #
            # Phase4Persistence is passed as:
            #
            #     store
            #         -> durable idempotency state
            #
            #     attempt_store
            #         -> durable pre-send order attempt state
            #
            # This preserves restart-safe behavior.
            # --------------------------------------------------------

            # Durably record the R denominator BEFORE the order can be
            # sent, keyed by execution identity, so restart recovery can
            # attach it to the broker position if the process dies after
            # the broker accepts the order.
            if sizing.dollar_risk is not None and sizing.dollar_risk > 0:
                persistence.record_attempt_risk(
                    encode_execution_identity(
                        build_execution_identity(event)
                    ),
                    sizing.dollar_risk,
                )

            result = guard_and_place_order(
                event,
                sizing,
                specs,
                order_config,
                store=persistence,
                attempt_store=persistence,
            )

            # --------------------------------------------------------
            # 10. ORDER RESULT
            # --------------------------------------------------------

            if result.accepted:
                print(
                    f"[order] accepted "
                    f"ticket={result.ticket} "
                    f"dry_run={result.dry_run}"
                )

                # The order ticket is NOT the position ID. Resolve the
                # broker position ID from deal history and record the
                # R denominator against it. If it cannot be confirmed,
                # nothing is recorded (no guessing) and the durable
                # order-attempt / recovery path handles the execution.
                record_risk_for_accepted_order(
                    connection,
                    result,
                    sizing.dollar_risk,
                    persistence,
                )

            else:
                print(
                    f"[order] rejected: {result.reason}"
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 4 live runtime loop."
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Submit real orders (dry_run=False). "
            "Without --live the runtime remains in dry-run mode."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Explicit no-op flag; dry-run is already the default."
        ),
    )

    args = parser.parse_args()

    if args.live and args.dry_run:
        raise SystemExit(
            "Cannot pass both --live and --dry-run."
        )

    run_live_loop(
        symbol="XAUUSD",
        timeframe="M5",
        terminal_path=(
            r"C:\Program Files\MetaTrader 5\terminal64.exe"
        ),
        db_path="phase_04_live.db",
        kill_switch_path=Path(
            "state/kill_switch.json"
        ),
        dry_run=not args.live,
        risk_config_path=Path(
            "state/paper_risk_config.json"
        ),
    )
