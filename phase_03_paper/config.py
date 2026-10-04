"""
phase_03_paper/config.py

Centralized Phase 3 settings.

DESIGN PRINCIPLE
----------------
This is the single source of truth for Phase 3 configuration.

Phase 3 modules should read configuration from this module instead
of scattering Phase 3 constants throughout the codebase.

This module does NOT redefine the existing strategy engine. Strategy
logic remains in the existing strategy modules.

IMPORTANT
---------
Some values below are Phase 3 defaults and may require reconciliation
with existing project configuration before they are considered final,
especially session windows and strategy parameters.

SAFETY
------
Phase 3 is PAPER TRADING ONLY.

SafetySettings intentionally does not expose setters or configurable
fields for switching execution into live mode.

Phase 4 will use a separate LiveExecutor implementation.

A configuration change must never be capable of enabling real orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time


# ============================================================
# GENERAL
# ============================================================


@dataclass(frozen=True)
class GeneralSettings:
    """General Phase 3 application settings."""

    system_name: str = "Machet Mechanics — Phase 3 Paper Trading"
    mode: str = "paper"
    timezone: str = "Africa/Nairobi"
    logging_level: str = "INFO"
    auto_start: bool = False
    paper_mode_confirmation_required: bool = True


# ============================================================
# MARKET DATA
# ============================================================


@dataclass(frozen=True)
class MarketDataSettings:
    """Market-data and candle-validation settings."""

    # Must match SYMBOL in phase_04_live/main.py.
    symbol: str = "XAUUSD"
    timeframe: str = "M5"

    # Phase 3 starts with a synthetic/demo source.
    #
    # This is intentionally pluggable. A future MT5, broker, or
    # market-data adapter can replace the source without requiring
    # changes to the downstream candle, strategy, or execution layers.
    data_source: str = "synthetic_demo"

    session_timezone: str = "Africa/Nairobi"

    # Candle older than this threshold is considered stale.
    stale_data_threshold_seconds: float = 60.0

    # Timestamps inside this tolerance are treated as duplicates.
    duplicate_timestamp_tolerance_seconds: float = 0.5


# ============================================================
# SESSIONS
# ============================================================


@dataclass(frozen=True)
class SessionWindow:
    """One configured trading-session window."""

    name: str
    start_local: time
    end_local: time
    enabled: bool = True


@dataclass(frozen=True)
class SessionSettings:
    """
    Central Phase 3 session configuration.

    NOTE:
    These session times are Phase 3 defaults and must be reconciled
    against the existing strategy/rules.py session definitions before
    they become the final source of truth for strategy filtering.

    The 10-minute warning is a required Phase 3 feature.
    """

    sessions: tuple[SessionWindow, ...] = field(
        default_factory=lambda: (
            SessionWindow(
                name="London",
                start_local=time(10, 0),
                end_local=time(19, 0),
            ),
            SessionWindow(
                name="New York",
                start_local=time(15, 0),
                end_local=time(0, 0),
            ),
        )
    )

    # Required Phase 3 behavior:
    # notify approximately 10 minutes before every enabled session.
    warning_minutes_before: int = 10


# ============================================================
# STRATEGY
# ============================================================


@dataclass(frozen=True)
class StrategySettings:
    """
    Phase 3 strategy parameters.

    These values are carried into the existing strategy/signal engine.
    This class does not implement strategy logic.
    """

    max_bars_to_retest: int = 20
    reward_multiple: float = 2.0
    stop_buffer: float = 0.0
    entry_mode: str = "midpoint"

    # Update once formal strategy versioning exists.
    strategy_version: str = "unspecified"


# ============================================================
# PAPER EXECUTION
# ============================================================


@dataclass(frozen=True)
class PaperExecutionSettings:
    """Simulation assumptions for Phase 3 paper execution."""

    starting_balance: float = 10_000.0

    # Simulated spread expressed in the instrument's price units.
    # XAUUSD: 0.30 = 30 cents. Replace with your broker's real
    # typical spread once measured in MT5 (keep it pessimistic).
    simulated_spread: float = 0.30

    # Supported models:
    #   none
    #   fixed
    #   random
    slippage_model: str = "fixed"

    # Maximum/fixed simulated adverse movement in price units.
    # XAUUSD: 0.10 = 10 cents. Placeholder until measured on real fills.
    slippage_fixed_amount: float = 0.10

    # Supported models:
    #   none
    #   fixed
    #   random
    latency_model: str = "fixed"

    # Simulated signal-to-fill delay.
    latency_fixed_ms: float = 150.0

    # Probability of simulated order rejection.
    #
    # 0.0 means rejection simulation is disabled.
    rejection_probability: float = 0.0


# ============================================================
# NOTIFICATIONS
# ============================================================


@dataclass(frozen=True)
class NotificationSettings:
    """
    Central notification configuration.

    Notification categories are handled by the notification engine.
    """

    session_notifications_enabled: bool = True
    trading_notifications_enabled: bool = True
    system_notifications_enabled: bool = True
    daily_ai_notifications_enabled: bool = True

    # Required Phase 3 session-warning timing.
    notification_warning_minutes_before: int = 10

    # Number of days notifications remain available.
    retention_days: int = 90


# ============================================================
# DASHBOARD
# ============================================================


@dataclass(frozen=True)
class DashboardSettings:
    """Phase 3 dashboard behavior."""

    refresh_interval_seconds: float = 5.0

    visible_metrics: tuple[str, ...] = (
        "equity",
        "daily_pnl",
        "daily_r",
        "win_rate",
        "drawdown",
        "open_position",
    )

    # Matches the professional dark dashboard theme.
    theme: str = "dark_multi_accent"

    # Show an unread notification indicator.
    show_unread_notification_badge: bool = True


# ============================================================
# AI
# ============================================================


@dataclass(frozen=True)
class AISettings:
    """
    Local Machet AI configuration.

    Phase 3 reuses the local Ollama architecture already established
    during Phase 2.

    The AI is an analysis/research assistant, not a trade executor.
    """

    local_model_id: str = "qwen2.5-coder:3b"

    enabled: bool = True

    # Maximum amount of contextual data supplied to the model.
    max_context_chars: int = 8000

    # Maximum number of recent conversation messages retained.
    max_chat_messages: int = 12

    # Maximum time allowed for an AI request.
    request_timeout_seconds: float = 120.0

    # Daily report generation is mandatory for the Phase 3 design.
    daily_report_enabled: bool = True

    # Daily report should produce a concise notification plus
    # a fuller report stored for later review.
    daily_notification_enabled: bool = True


# ============================================================
# PERSISTENCE
# ============================================================


@dataclass(frozen=True)
class PersistenceSettings:
    """State and journal persistence settings."""

    enabled: bool = True

    # Phase 3 data directory.
    data_directory: str = "phase_03_paper/data"

    # Persist open positions/orders so restart recovery is possible.
    persist_open_state: bool = True

    # Persist the complete paper-trade journal.
    persist_trade_journal: bool = True

    # Persist notifications.
    persist_notifications: bool = True

    # Persist audit events.
    persist_audit_events: bool = True

    # Persist daily AI reports.
    persist_ai_reports: bool = True


# ============================================================
# MONITORING
# ============================================================


@dataclass(frozen=True)
class MonitoringSettings:
    """System-health and watchdog configuration."""

    enabled: bool = True

    # Watchdog polling interval.
    watchdog_interval_seconds: float = 5.0

    # Engine should warn when market data becomes too old.
    monitor_data_freshness: bool = True

    # Monitor execution engine health.
    monitor_execution: bool = True

    # Monitor notification engine.
    monitor_notifications: bool = True

    # Monitor AI report generation.
    monitor_ai: bool = True

    # Monitor persistence/recovery state.
    monitor_persistence: bool = True


# ============================================================
# SAFETY
# ============================================================


@dataclass(frozen=True)
class SafetySettings:
    """
    Non-configurable Phase 3 safety boundary.

    These properties deliberately cannot be changed through a normal
    settings object.

    Phase 3 must never be able to send real broker orders.
    """

    @property
    def paper_mode(self) -> bool:
        """Phase 3 is permanently paper mode."""
        return True

    @property
    def real_orders_disabled(self) -> bool:
        """Real orders are permanently disabled in Phase 3."""
        return True


# ============================================================
# COMPLETE PHASE 3 CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class Phase3Config:
    """
    Complete centralized Phase 3 configuration.

    All Phase 3 components should receive this configuration rather
    than independently defining their own constants.
    """

    general: GeneralSettings = field(default_factory=GeneralSettings)

    market_data: MarketDataSettings = field(
        default_factory=MarketDataSettings
    )

    sessions: SessionSettings = field(
        default_factory=SessionSettings
    )

    strategy: StrategySettings = field(
        default_factory=StrategySettings
    )

    paper_execution: PaperExecutionSettings = field(
        default_factory=PaperExecutionSettings
    )

    notifications: NotificationSettings = field(
        default_factory=NotificationSettings
    )

    dashboard: DashboardSettings = field(
        default_factory=DashboardSettings
    )

    ai: AISettings = field(
        default_factory=AISettings
    )

    persistence: PersistenceSettings = field(
        default_factory=PersistenceSettings
    )

    monitoring: MonitoringSettings = field(
        default_factory=MonitoringSettings
    )

    safety: SafetySettings = field(
        default_factory=SafetySettings
    )


# ============================================================
# DEFAULT CONFIGURATION INSTANCE
# ============================================================


DEFAULT_CONFIG = Phase3Config()


# ============================================================
# VALIDATION
# ============================================================


def validate_config(config: Phase3Config = DEFAULT_CONFIG) -> None:
    """
    Validate Phase 3 configuration.

    Raises:
        ValueError: if a configuration violates Phase 3 requirements.
    """

    if config.general.mode != "paper":
        raise ValueError(
            "Phase 3 must use paper mode. "
            f"Received mode={config.general.mode!r}."
        )

    if not config.safety.paper_mode:
        raise ValueError(
            "Phase 3 safety violation: paper_mode must be True."
        )

    if not config.safety.real_orders_disabled:
        raise ValueError(
            "Phase 3 safety violation: real orders must remain disabled."
        )

    if config.sessions.warning_minutes_before != 10:
        raise ValueError(
            "Phase 3 requires a 10-minute session warning."
        )

    if config.notifications.notification_warning_minutes_before != 10:
        raise ValueError(
            "Phase 3 notification warning must be 10 minutes."
        )

    if config.market_data.stale_data_threshold_seconds <= 0:
        raise ValueError(
            "stale_data_threshold_seconds must be greater than zero."
        )

    if config.paper_execution.starting_balance <= 0:
        raise ValueError(
            "starting_balance must be greater than zero."
        )

    if not 0.0 <= config.paper_execution.rejection_probability <= 1.0:
        raise ValueError(
            "rejection_probability must be between 0.0 and 1.0."
        )

    if config.paper_execution.slippage_model not in {
        "none",
        "fixed",
        "random",
    }:
        raise ValueError(
            "slippage_model must be 'none', 'fixed', or 'random'."
        )

    if config.paper_execution.latency_model not in {
        "none",
        "fixed",
        "random",
    }:
        raise ValueError(
            "latency_model must be 'none', 'fixed', or 'random'."
        )

    if config.notifications.retention_days <= 0:
        raise ValueError(
            "notification retention_days must be greater than zero."
        )

    if config.dashboard.refresh_interval_seconds <= 0:
        raise ValueError(
            "dashboard refresh interval must be greater than zero."
        )


# ============================================================
# PUBLIC HELPER
# ============================================================


def get_default_config() -> Phase3Config:
    """
    Return the validated default Phase 3 configuration.
    """

    validate_config(DEFAULT_CONFIG)
    return DEFAULT_CONFIG


__all__ = [
    "GeneralSettings",
    "MarketDataSettings",
    "SessionWindow",
    "SessionSettings",
    "StrategySettings",
    "PaperExecutionSettings",
    "NotificationSettings",
    "DashboardSettings",
    "AISettings",
    "PersistenceSettings",
    "MonitoringSettings",
    "SafetySettings",
    "Phase3Config",
    "DEFAULT_CONFIG",
    "validate_config",
    "get_default_config",
]
