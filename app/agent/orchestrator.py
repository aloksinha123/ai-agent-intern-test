from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

from app.agent.router import RouteDecision, Router
from app.agent.session import SessionContext
from app.llm.client import GeminiClient
from app.observability.trace import (
    AgentTrace,
    RetrievalDiagnostic,
    ToolCallDiagnostic,
)
from app.orders.service import OrderService
from app.rag.conflict import analyze_evidence
from app.rag.vector_store import VectorStore


@dataclass(frozen=True)
class AgentTurnResponse:
    """Structured response from a single agent conversational turn."""

    session_id: str
    intent: str
    text: str
    citations: List[str] = field(default_factory=list)
    handoff_required: bool = False
    order_id_used: Optional[str] = None
    evidence_status: Optional[str] = None
    tool_called: bool = False
    tool_name: Optional[str] = None
    route_reason: Optional[str] = None
    trace: Optional[AgentTrace] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent response for observability and callers."""
        d = asdict(self)
        if self.trace:
            d["trace"] = self.trace.to_dict()
        return d


class AgentOrchestrator:
    """Deterministic orchestrator managing sessions, routing, RAG retrieval, order lookup, and Gemini LLM."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        gemini_client: Optional[GeminiClient] = None,
        order_service: Optional[OrderService] = None,
        enable_tracing: bool = False,
    ) -> None:
        """Initialize orchestrator with components.

        Args:
            vector_store: Optional loaded VectorStore. If None, loads from default persisted index.
            gemini_client: Optional GeminiClient. If None, instantiates default GeminiClient.
            order_service: Optional OrderService. If None, instantiates default OrderService.
            enable_tracing: If True, produces AgentTrace for every turn by default.
        """
        self.vector_store = vector_store
        self.gemini_client = gemini_client
        self.order_service = order_service
        self.enable_tracing = enable_tracing
        self.sessions: Dict[str, SessionContext] = {}

    def _ensure_components(self) -> None:
        """Lazy load default components if not provided during init."""
        if self.vector_store is None:
            self.vector_store = VectorStore.load()
        if self.gemini_client is None:
            self.gemini_client = GeminiClient()
        if self.order_service is None:
            self.order_service = OrderService()

    def get_or_create_session(self, session_id: str) -> SessionContext:
        """Retrieve existing SessionContext or create a new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionContext(session_id=session_id)
        return self.sessions[session_id]

    def process_turn(
        self,
        message: str,
        session_id: str = "default",
        enable_tracing: Optional[bool] = None,
    ) -> AgentTurnResponse:
        """Execute a full conversational turn with deterministic routing and application authority.

        Args:
            message: User's customer inquiry.
            session_id: Unique session identifier for multi-turn state.
            enable_tracing: Optional override for per-turn trace collection.

        Returns:
            AgentTurnResponse object.
        """
        collect_trace = self.enable_tracing if enable_tracing is None else enable_tracing

        self._ensure_components()
        session = self.get_or_create_session(session_id)

        # 1. Deterministic Routing
        decision: RouteDecision = Router.route(query=message, session=session)

        response_text = ""
        citations: List[str] = []
        handoff_required = False
        order_id_used: Optional[str] = None
        evidence_status: Optional[str] = None
        tool_called = False
        tool_name: Optional[str] = None
        fallback_reason: Optional[str] = None
        retrieval_diagnostics: List[RetrievalDiagnostic] = []
        tool_diagnostic: Optional[ToolCallDiagnostic] = None
        model_name: Optional[str] = getattr(self.gemini_client, "model_name", "gemini-3.6-flash")

        # 2. Intent Dispatch
        if decision.intent == "greeting":
            response_text = (
                "Hello! How can I assist you with your Aster & Row questions or orders today?"
            )
            handoff_required = False
            session.set_topic("greeting")

        elif decision.intent == "unknown":
            response_text = (
                "I'm sorry, I didn't quite understand that. Could you please clarify if you "
                "have a question about our policies, products, or need help looking up an order?"
            )
            handoff_required = False
            fallback_reason = "unrecognized_intent"

        elif decision.intent == "order":
            if decision.reason == "malformed_order_id":
                response_text = (
                    f"The order ID '{decision.order_id}' appears to be invalid. Aster & Row "
                    "order IDs follow the format ORD-XXXX (e.g., ORD-1001). Please check and "
                    "provide a valid order ID."
                )
                handoff_required = False
                fallback_reason = "malformed_order_id"

            elif decision.normalized_order_id is None:
                # Missing order ID in query and no session context
                response_text = (
                    "I'd be happy to help check your order status! Please provide your order ID "
                    "(format: ORD-XXXX, such as ORD-1001)."
                )
                handoff_required = False
                fallback_reason = "missing_order_id"

            else:
                # Execute deterministic order lookup tool
                norm_id = decision.normalized_order_id
                order_res = self.order_service.lookup(norm_id)
                tool_called = True
                tool_name = "order_lookup"
                order_id_used = norm_id

                if not order_res.found:
                    response_text = (
                        f"I could not find an order matching '{norm_id}' in our records. "
                        "Please double-check your order number. I recommend connecting with a "
                        "support specialist for further assistance."
                    )
                    handoff_required = bool(order_res.handoff_required)
                    fallback_reason = "unknown_order"
                    if collect_trace:
                        tool_diagnostic = ToolCallDiagnostic(
                            tool_name="order_lookup",
                            normalized_order_id=norm_id,
                            success=False,
                            handoff_required=handoff_required,
                            sanitized_field_summary={"found": False, "reason": order_res.error_message},
                        )
                else:
                    # Update session order state
                    session.set_order_context(norm_id)
                    order_data = order_res.order.to_dict()
                    
                    # Action or Privacy request check: asking for unsupported actions (like cancel) or private fields requires handoff
                    is_action_or_privacy_request = bool(
                        re.search(
                            r"\b(email|address|shipping\s*address|risk\s*score|internal\s*note|warehouse\s*note|support\s*tags|cancel|cancellation|refund|change\s*address|modify)\b",
                            message,
                            re.I,
                        )
                    )
                    handoff_required = bool(order_res.handoff_required) or is_action_or_privacy_request
                    if is_action_or_privacy_request:
                        fallback_reason = "unsupported_action_or_privacy_request"

                    if collect_trace:
                        tool_diagnostic = ToolCallDiagnostic(
                            tool_name="order_lookup",
                            normalized_order_id=norm_id,
                            success=True,
                            handoff_required=handoff_required,
                            sanitized_field_summary={
                                "status": str(order_res.order.status),
                                "item_count": len(order_res.order.items),
                                "has_carrier": bool(order_res.order.carrier),
                                "has_eta": bool(order_res.order.estimated_delivery),
                                "is_processing": order_res.order.status == "processing",
                            },
                        )

                    # Synthesize natural response via Gemini with untrusted tool data
                    llm_resp = self.gemini_client.generate_order_response(
                        query=message,
                        order_data=order_data,
                        handoff_required=handoff_required,
                    )
                    response_text = llm_resp.text
                    # Strict application handoff authority
                    handoff_required = bool(order_res.handoff_required) or is_action_or_privacy_request

        elif decision.intent == "knowledge":
            query_to_retrieve = decision.contextual_query or message
            session.set_topic(decision.topic or "general", entity=decision.entity)

            # RAG Retrieval
            evidence = self.vector_store.retrieve(query=query_to_retrieve, top_k=5, mode="customer")

            if collect_trace:
                for chunk in evidence:
                    retrieval_diagnostics.append(
                        RetrievalDiagnostic(
                            filename=chunk.filename,
                            heading=chunk.heading,
                            semantic_score=float(chunk.score),
                            final_score=float(chunk.score),
                            policy_eligibility=True,
                            policy_reason="official_active_customer",
                        )
                    )

            # Deterministic Evidence & Conflict Analysis
            analysis = analyze_evidence(results=evidence, query=query_to_retrieve)
            evidence_status = analysis.status
            handoff_required = bool(analysis.handoff_required)

            if analysis.status == "conflict":
                fallback_reason = "evidence_conflict_detected"
            elif analysis.status == "insufficient":
                fallback_reason = "insufficient_evidence"

            # LLM Generation with validated citations
            llm_resp = self.gemini_client.generate_response(
                query=message,
                evidence=evidence,
                analysis=analysis,
            )
            response_text = llm_resp.text
            citations = llm_resp.citations
            # Strict application handoff authority
            handoff_required = bool(analysis.handoff_required)

        # 3. Update Multi-Turn Session History
        session.add_turn("user", message)
        session.add_turn("assistant", response_text)

        # 4. Optional Structured Trace Generation
        trace_obj: Optional[AgentTrace] = None
        if collect_trace:
            trace_obj = AgentTrace(
                session_id=session.session_id,
                user_message=message,
                route_intent=decision.intent,
                route_reason=decision.reason or "default",
                order_id_detected=decision.order_id,
                retrieved_results=retrieval_diagnostics,
                evidence_analysis=evidence_status,
                tool_call=tool_diagnostic,
                sanitized_tool_result_summary=tool_diagnostic.sanitized_field_summary if tool_diagnostic else None,
                model_name=model_name,
                final_response=response_text,
                citations=citations,
                handoff_required=handoff_required,
                fallback_reason=fallback_reason,
            )

        return AgentTurnResponse(
            session_id=session.session_id,
            intent=decision.intent,
            text=response_text,
            citations=citations,
            handoff_required=handoff_required,
            order_id_used=order_id_used,
            evidence_status=evidence_status,
            tool_called=tool_called,
            tool_name=tool_name,
            route_reason=decision.reason,
            trace=trace_obj,
        )
