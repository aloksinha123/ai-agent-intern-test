"""Lightweight structured tracing and observability for Aster & Row support agent.

This module provides privacy-safe, structured diagnostics for routing, retrieval,
tool execution, and fallback paths. It strictly excludes PII, secrets, internal notes,
risk scores, and full document dumps.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class RetrievalDiagnostic:
    """Safe diagnostic metadata for a single retrieved chunk."""

    filename: str
    heading: str
    semantic_score: float
    final_score: float
    policy_eligibility: bool
    policy_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "heading": self.heading,
            "semantic_score": round(self.semantic_score, 4),
            "final_score": round(self.final_score, 4),
            "policy_eligibility": self.policy_eligibility,
            "policy_reason": self.policy_reason,
        }


@dataclass
class ToolCallDiagnostic:
    """Sanitized diagnostic summary for tool invocations."""

    tool_name: str
    normalized_order_id: Optional[str]
    success: bool
    handoff_required: bool
    sanitized_field_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "normalized_order_id": self.normalized_order_id,
            "success": self.success,
            "handoff_required": self.handoff_required,
            "sanitized_field_summary": self.sanitized_field_summary,
        }


@dataclass
class AgentTrace:
    """Structured, privacy-safe audit trace for a single agent turn."""

    session_id: str
    user_message: str
    route_intent: str
    route_reason: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    order_id_detected: Optional[str] = None
    retrieved_results: List[RetrievalDiagnostic] = field(default_factory=list)
    evidence_analysis: Optional[str] = None
    tool_call: Optional[ToolCallDiagnostic] = None
    sanitized_tool_result_summary: Optional[Dict[str, Any]] = None
    model_name: Optional[str] = None
    final_response: str = ""
    citations: List[str] = field(default_factory=list)
    handoff_required: bool = False
    error: Optional[str] = None
    fallback_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "user_message": self.user_message,
            "route_intent": self.route_intent,
            "route_reason": self.route_reason,
            "order_id_detected": self.order_id_detected,
            "retrieved_results": [r.to_dict() for r in self.retrieved_results],
            "evidence_analysis": self.evidence_analysis,
            "tool_call": self.tool_call.to_dict() if self.tool_call else None,
            "sanitized_tool_result_summary": self.sanitized_tool_result_summary,
            "model_name": self.model_name,
            "final_response": self.final_response,
            "citations": self.citations,
            "handoff_required": self.handoff_required,
            "error": self.error,
            "fallback_reason": self.fallback_reason,
        }
