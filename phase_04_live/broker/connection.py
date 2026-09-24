"""
phase_04_live/broker/connection.py

Explicit MT5 connection lifecycle for Phase 4's live broker layer.

Purpose
-------
BrokerConnection owns the local connection lifecycle used by Phase 4
broker-facing components.

It is intentionally separate from:

    phase_04_live/market/live_source.py

LiveMarketSource owns market-data concerns such as symbol selection
and candle polling. BrokerConnection owns the broker connection
lifecycle only.

This class does NOT:
    - select symbols
    - read account state
    - read positions
    - read orders
    - place orders
    - perform reconciliation
    - make risk decisions
    - determine whether the broker is operational beyond the
      result of the MT5 initialize() call


Connection-state rule
---------------------
is_connected() reports this object's LOCAL connection state.

It does NOT perform an MT5 round-trip.

Therefore:

    is_connected() == True

means:

    "this BrokerConnection successfully initialized MT5 and has
     not locally disconnected."

It does NOT guarantee that MT5 or the broker is still reachable.

Components that require current broker health must perform an
actual broker read and handle failure explicitly.


Architecture
------------
    BrokerConnection
          |
          +--> AccountStateProvider
          +--> PositionStateProvider
          +--> OrderStateProvider
          +--> Reconciliation
          |
          +--> Live runtime / monitoring

Market-data responsibilities remain in LiveMarketSource.


Important lifecycle rule
------------------------
MetaTrader5 exposes a process-level terminal connection.

Therefore Phase 4 should avoid treating individual components as
independent owners of that global connection.

The runtime coordinator should control startup/shutdown ordering
so that one component does not disconnect MT5 while another
component still requires it.
"""

from __future__ import annotations

import MetaTrader5 as mt5


MT5_TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"


class BrokerConnection:
    """
    Explicit lifecycle wrapper around the MT5 terminal connection.

    Example:

        connection = BrokerConnection()

        connection.connect()

        if connection.is_connected():
            # Perform broker reads here.

        connection.disconnect()

    is_connected() is intentionally a cheap local-state check.

    It must not be treated as proof that the broker is currently
    reachable. Current broker health should be established by an
    actual MT5 operation performed by the appropriate reader/provider.
    """

    def __init__(
        self,
        terminal_path: str = MT5_TERMINAL_PATH,
    ) -> None:
        self.terminal_path = terminal_path
        self._connected = False

    def connect(self) -> None:
        """
        Initialize the MT5 terminal connection.

        If this instance already considers itself connected, the
        method is idempotent and returns without performing another
        initialization call.

        Raises:
            RuntimeError:
                If MT5 initialization fails.
        """

        if self._connected:
            return

        if not mt5.initialize(path=self.terminal_path):
            self._connected = False
            raise RuntimeError(
                f"MT5 initialize() failed: {mt5.last_error()}"
            )

        self._connected = True

    def disconnect(self) -> None:
        """
        Shut down the MT5 connection owned by this lifecycle object.

        The local state is cleared regardless of the shutdown path.
        """

        if not self._connected:
            return

        try:
            mt5.shutdown()
        finally:
            self._connected = False

    def is_connected(self) -> bool:
        """
        Return this object's local connection state.

        This is NOT a live broker-health check.

        A True result only means that this instance successfully
        initialized MT5 and has not locally disconnected since then.
        """
        return self._connected