"""
phase_04_live/execution/consumer.py

Signal consumption pipeline: LiveEventEngine events -> ExecutionRequests -> BrokerExecutor.

This thin layer sits between the event engine and the broker executor:
    1. Listens for SIGNAL_TRIGGERED events
    2. Transforms them into ExecutionRequests
    3. Submits to BrokerExecutor
    4. Records ExecutionResults back into the event stream for monitoring

This is NOT strategy logic. It is NOT execution logic. It is integration scaffolding.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, List, Optional

from phase_03_paper.signals.adapter import AdapterSignalEvent
from phase_04_live.events.event_types import LiveEvent, LiveEventType
from phase_04_live.execution.executor import BrokerExecutor
from phase_04_live.execution.models import ExecutionRequest, ExecutionResult


logger = logging.getLogger(__name__)


class ExecutionEventConsumer:
    """
    Convert live events into execution requests and submit to broker.
    
    Usage:
        consumer = ExecutionEventConsumer(broker_executor)
        
        # After engine.run_once() produces events:
        events = engine.run_once()
        execution_events = consumer.consume(events)
        
        # execution_events now contains the original events plus
        # ORDER_PLACED events for any SIGNAL_TRIGGERED that was executed.
    """

    def __init__(
        self,
        broker_executor: BrokerExecutor,
        symbol: str = "XAUUSD",
    ):
        self._broker_executor = broker_executor
        self._symbol = symbol

    def consume(self, events: List[LiveEvent]) -> List[LiveEvent]:
        """
        Process live events and execute any triggered signals.
        
        Returns the original events plus new ORDER_PLACED/ORDER_REJECTED events.
        """
        execution_events: List[LiveEvent] = []
        
        for event in events:
            execution_events.append(event)
            
            # Only process SIGNAL_TRIGGERED events
            if event.event_type != LiveEventType.SIGNAL_TRIGGERED:
                continue
            
            try:
                result = self._execute_signal(event)
                if result is not None:
                    execution_events.append(result)
            except Exception as exc:
                logger.error(
                    "[CONSUMER] Unhandled error processing signal event: %s",
                    exc,
                    exc_info=True,
                )
        
        return execution_events

    def _execute_signal(self, event: LiveEvent) -> Optional[LiveEvent]:
        """
        Transform a SIGNAL_TRIGGERED event into an execution request
        and submit to broker.
        
        Returns a new LiveEvent wrapping the ExecutionResult, or None
        if the event payload is malformed.
        """
        
        # Unpack the signal
        if not isinstance(event.payload, AdapterSignalEvent):
            logger.warning(
                "[CONSUMER] Ignoring SIGNAL_TRIGGERED with non-AdapterSignalEvent payload: %s",
                type(event.payload),
            )
            return None
        
        signal_event: AdapterSignalEvent = event.payload
        
        # ============================================================
        # Transform signal into execution request
        # ============================================================
        
        signal_id = self._make_signal_id(signal_event)
        
        # Determine order direction from signal (the adapter defines this)
        order_type = self._infer_order_type(signal_event)
        if order_type is None:
            logger.warning(
                "[CONSUMER] Could not infer order type from signal: %s",
                signal_event,
            )
            return None
        
        request = ExecutionRequest(
            signal_id=signal_id,
            symbol=self._symbol,
            order_type=order_type,
            entry_price=float(signal_event.fill_price),
            initial_risk=float(signal_event.initial_risk),
            signal_generated_at=event.occurred_at,
            request_created_at=datetime.now(timezone.utc),
        )
        
        # ============================================================
        # Submit to executor
        # ============================================================
        
        logger.info(
            "[CONSUMER] Executing %s signal: %s | price=%.2f | risk=%.2f",
            order_type,
            signal_id,
            request.entry_price,
            request.initial_risk,
        )
        
        result = self._broker_executor.execute(request)
        
        # ============================================================
        # Return execution result as a live event
        # ============================================================
        
        event_type = (
            LiveEventType.ORDER_PLACED
            if result.succeeded
            else LiveEventType.ORDER_REJECTED
        )
        
        return LiveEvent(
            event_type=event_type,
            occurred_at=result.executed_at,
            payload=result,
        )

    def _make_signal_id(self, signal_event: AdapterSignalEvent) -> str:
        """
        Create a stable signal identifier from the adapter event.
        
        This ID is used for deduplication and audit trails.
        """
        return (
            f"signal_{signal_event.setup_bar_index}_"
            f"{signal_event.trigger_bar_index}_"
            f"{signal_event.signal.signal_type if hasattr(signal_event.signal, 'signal_type') else 'unknown'}"
        )

    @staticmethod
    def _infer_order_type(signal_event: AdapterSignalEvent) -> Optional[str]:
        """
        Infer BUY/SELL from the signal's direction.
        
        Adapter signals carry direction info; extract it here.
        This is adapter-specific and may need to change as the adapter evolves.
        """
        signal = signal_event.signal
        
        # Try to infer from signal attributes
        if hasattr(signal, "direction"):
            direction = getattr(signal, "direction", None)
            if direction and direction.upper() in ("LONG", "BUY", "UP"):
                return "BUY"
            elif direction and direction.upper() in ("SHORT", "SELL", "DOWN"):
                return "SELL"
        
        if hasattr(signal, "signal_type"):
            signal_type = getattr(signal, "signal_type", "").upper()
            if "BUY" in signal_type or "LONG" in signal_type:
                return "BUY"
            elif "SELL" in signal_type or "SHORT" in signal_type:
                return "SELL"
        
        # Fallback: unable to determine
        logger.warning(
            "[CONSUMER] Could not infer order direction from signal attributes: %s",
            signal,
        )
        return None


__all__ = ["ExecutionEventConsumer"]
