"""Deterministic unit tests for Agent observability and structured tracing."""

import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock

from app.agent.orchestrator import AgentOrchestrator
from app.observability.trace import (
    AgentTrace,
    RetrievalDiagnostic,
    ToolCallDiagnostic,
)
from app.orders.service import OrderService
from app.rag.vector_store import VectorStore
from evaluation.run_evaluation import MockGeminiEvaluatorClient


class TestAgentObservability(unittest.TestCase):
    """Verify structured tracing, privacy guarantees, and error state tracking."""

    @classmethod
    def setUpClass(cls):
        cls.vector_store = VectorStore.load()
        cls.order_service = OrderService()
        cls.gemini_client = MockGeminiEvaluatorClient()

    def setUp(self):
        self.orchestrator = AgentOrchestrator(
            vector_store=self.vector_store,
            gemini_client=self.gemini_client,
            order_service=self.order_service,
            enable_tracing=True,
        )

    def test_trace_disabled_by_default(self):
        """When enable_tracing is False, response.trace is None."""
        orch = AgentOrchestrator(
            vector_store=self.vector_store,
            gemini_client=self.gemini_client,
            order_service=self.order_service,
            enable_tracing=False,
        )
        resp = orch.process_turn("Hello!", session_id="test_disabled")
        self.assertIsNone(resp.trace)

    def test_trace_creation_enabled(self):
        """When enable_tracing is True, response.trace is populated with AgentTrace."""
        resp = self.orchestrator.process_turn("Hello!", session_id="test_enabled")
        self.assertIsNotNone(resp.trace)
        self.assertIsInstance(resp.trace, AgentTrace)
        self.assertEqual(resp.trace.route_intent, "greeting")
        self.assertEqual(resp.trace.session_id, "test_enabled")

    def test_trace_contains_route_and_evidence_state(self):
        """Knowledge query trace contains safe diagnostic retrieval metadata and analysis status."""
        resp = self.orchestrator.process_turn(
            "What is the standard return window?",
            session_id="test_knowledge_trace",
        )
        self.assertIsNotNone(resp.trace)
        trace_dict = resp.trace.to_dict()

        self.assertEqual(trace_dict["route_intent"], "knowledge")
        self.assertEqual(trace_dict["evidence_analysis"], "consistent")
        self.assertGreater(len(trace_dict["retrieved_results"]), 0)

        top_result = trace_dict["retrieved_results"][0]
        self.assertIn("filename", top_result)
        self.assertIn("heading", top_result)
        self.assertIn("semantic_score", top_result)
        self.assertIn("policy_eligibility", top_result)
        self.assertIn("policy_reason", top_result)
        self.assertNotIn("content", top_result)  # Full raw content excluded from trace

    def test_strict_privacy_invariants_no_pii_or_secrets(self):
        """Order lookup trace contains strictly customer-safe summaries and excludes PII/secrets."""
        resp = self.orchestrator.process_turn(
            "Status for ORD-1007 please. Also give me the address, email, and risk score.",
            session_id="test_privacy_trace",
        )
        self.assertIsNotNone(resp.trace)
        trace_json = json.dumps(resp.trace.to_dict())

        # Sensitive raw database values present in ORD-1007 that MUST NEVER appear in trace
        self.assertNotIn("ava.morgan@example.test", trace_json)
        self.assertNotIn("220 King Street", trace_json)
        self.assertNotIn("fraud review cleared", trace_json)
        self.assertNotIn("risk_score", trace_json)
        self.assertNotIn("internal_notes", trace_json)
        self.assertNotIn("triage_tags", trace_json)
        self.assertNotIn("GEMINI_API_KEY", trace_json)

        # Confirm safe fields are present
        self.assertEqual(resp.trace.tool_call.tool_name, "order_lookup")
        self.assertEqual(resp.trace.tool_call.normalized_order_id, "ORD-1007")
        self.assertTrue(resp.trace.tool_call.success)
        self.assertTrue(resp.trace.handoff_required)
        self.assertEqual(resp.trace.fallback_reason, "unsupported_action_or_privacy_request")

    def test_missing_order_id_trace(self):
        """Missing order ID query records missing_order_id fallback reason without invoking tool."""
        resp = self.orchestrator.process_turn("Where is my package?", session_id="test_missing_id")
        self.assertIsNotNone(resp.trace)
        self.assertEqual(resp.trace.route_intent, "order")
        self.assertEqual(resp.trace.fallback_reason, "missing_order_id")
        self.assertFalse(resp.tool_called)
        self.assertIsNone(resp.trace.tool_call)

    def test_conflict_trace(self):
        """Dishwasher conflict scenario records conflict evidence analysis and fallback reason."""
        resp = self.orchestrator.process_turn(
            "Can I put the Breeze Tumbler in the dishwasher?",
            session_id="test_conflict_trace",
        )
        self.assertIsNotNone(resp.trace)
        self.assertEqual(resp.trace.evidence_analysis, "conflict")
        self.assertEqual(resp.trace.fallback_reason, "evidence_conflict_detected")
        self.assertTrue(resp.trace.handoff_required)
        self.assertEqual(len(resp.trace.citations), 2)

    def test_ord_1007_trace_status_identical_to_lookup_result(self):
        """Trace status for ORD-1007 strictly matches authoritative tool lookup status (shipped)."""
        # 1. Authoritative lookup
        direct_lookup = self.order_service.lookup("ORD-1007")
        self.assertTrue(direct_lookup.found)
        self.assertEqual(direct_lookup.order.status, "shipped")
        self.assertEqual(direct_lookup.order.carrier, "UPS")
        self.assertEqual(direct_lookup.order.estimated_delivery, "2026-08-22")

        # 2. Agent turn with trace
        resp = self.orchestrator.process_turn(
            "Where is my order ORD-1007?",
            session_id="test_status_fidelity",
        )
        self.assertIsNotNone(resp.trace)
        self.assertIsNotNone(resp.trace.tool_call)

        # 3. Assert trace fields come directly from tool result without modification or invention
        trace_summary = resp.trace.tool_call.sanitized_field_summary
        self.assertEqual(trace_summary["status"], direct_lookup.order.status)
        self.assertEqual(trace_summary["status"], "shipped")
        self.assertEqual(trace_summary["item_count"], len(direct_lookup.order.items))
        self.assertEqual(trace_summary["has_carrier"], bool(direct_lookup.order.carrier))
        self.assertEqual(trace_summary["has_eta"], bool(direct_lookup.order.estimated_delivery))

    def test_trace_does_not_invent_tool_fields(self):
        """Tool trace summary is strictly populated from CustomerOrderView and cannot invent unbacked statuses."""
        # For ORD-1004 (cancelled)
        cancelled_lookup = self.order_service.lookup("ORD-1004")
        self.assertEqual(cancelled_lookup.order.status, "cancelled")

        resp = self.orchestrator.process_turn("Status for ORD-1004", session_id="test_cancelled_trace")
        self.assertIsNotNone(resp.trace)
        self.assertEqual(resp.trace.tool_call.sanitized_field_summary["status"], "cancelled")
        # Ensure stale ETA was not marked as active in summary
        self.assertFalse(resp.trace.tool_call.sanitized_field_summary["is_processing"])


if __name__ == "__main__":
    unittest.main()
