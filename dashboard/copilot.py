from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CopilotResponse:
    answer: str
    metadata: Dict[str, Any]


class KulechoCopilotService:
    """Simple dashboard-facing service for Kulecho AI questions."""

    def __init__(self, query_engine) -> None:
        self.query_engine = query_engine

    def ask(
        self,
        question: str,
        *,
        static_knowledge: Optional[Dict[str, Any]] = None,
        current_state: Optional[Dict[str, Any]] = None,
        historical_context: Optional[Dict[str, Any]] = None,
        chart_context: Optional[Dict[str, Any]] = None,
    ) -> CopilotResponse:
        answer = self.query_engine.ask(
            question=question,
            static_knowledge=static_knowledge,
            current_state=current_state,
            historical_context=historical_context,
            chart_context=chart_context,
        )
        return CopilotResponse(
            answer=answer,
            metadata={
                "question": question,
                "static_knowledge": bool(static_knowledge),
                "current_state": bool(current_state),
                "historical_context": bool(historical_context),
                "chart_context": bool(chart_context),
            },
        )


__all__ = ["CopilotResponse", "KulechoCopilotService"]
