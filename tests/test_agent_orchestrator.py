"""Unit and integration tests for AgentOrchestrator."""

import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from app.agent.orchestrator import AgentOrchestrator, AgentTurnResponse
from app.llm.client import GeminiClient, LLMResponse
from app.orders.service import OrderService
from app.rag.conflict import EvidenceAnalysis
from app.rag.models import SearchResult
from app.rag.vector_store import VectorStore


class TestAgentOrchestrator(unittest.TestCase):
    """Test full agent orchestration flow with mocked LLM."""

    def setUp(self):
        # Create a mock vector store
        self.mock_vector_store = MagicMock(spec=VectorStore)

        # Create a mock Gemini client
        self.mock_gemini_client = MagicMock(spec=GeminiClient)

        # Use real OrderService pointing to data/orders.json
        self.order_service = OrderService()

        # Initialize orchestrator with mocked components
        self.orchestrator = AgentOrchestrator(
            vector_store=self.mock_vector_store,
            gemini_client=self.mock_gemini_client,
            order_service=self.order_service,
        )

        # Standard dummy search result
        self.sample_chunk = SearchResult(
            chunk_id="doc_01_chunk_01",
            filename="01-returns-policy-current.md",
            document_title="Return Policy",
            heading="Standard return window",
            content="Standard return window is 30 calendar days from delivery.",
            metadata={"status": "active", "policy_authority": "official_policy", "audience": "customer"},
            policy_eligible=True,
            policy_reason="active",
            score=0.88,
            final_score=0.88,
        )

    def test_greeting_does_not_call_rag_or_order_tool(self):
        """Greeting should return polite greeting without calling RAG or order tools."""
        resp = self.orchestrator.process_turn("Hello there!", session_id="s_greet")
        self.assertEqual(resp.intent, "greeting")
        self.assertFalse(resp.tool_called)
        self.assertFalse(resp.handoff_required)
        self.assertIn("Hello", resp.text)
        self.mock_vector_store.retrieve.assert_not_called()

    def test_unknown_asks_clarification(self):
        """Ambiguous/unknown input asks for clarification without calling tools."""
        resp = self.orchestrator.process_turn("xyz987qwe123", session_id="s_unk")
        self.assertEqual(resp.intent, "unknown")
        self.assertFalse(resp.tool_called)
        self.assertIn("clarify", resp.text.lower())
        self.mock_vector_store.retrieve.assert_not_called()

    def test_knowledge_question_calls_rag_and_gemini(self):
        """Knowledge query executes RAG retrieval, analysis, and Gemini generation."""
        self.mock_vector_store.retrieve.return_value = [self.sample_chunk]
        self.mock_gemini_client.generate_response.return_value = LLMResponse(
            text="The standard return window is 30 calendar days.\n\nSources:\n- 01-returns-policy-current.md — Standard return window",
            model_name="mock-model",
            analysis_status="consistent",
            handoff_required=False,
            evidence_count=1,
            evidence_chunk_ids=["doc_01_chunk_01"],
            citations=["01-returns-policy-current.md — Standard return window"],
        )

        resp = self.orchestrator.process_turn("What is the standard return window?", session_id="s_know")
        self.assertEqual(resp.intent, "knowledge")
        self.assertFalse(resp.tool_called)
        self.assertEqual(resp.evidence_status, "consistent")
        self.assertFalse(resp.handoff_required)
        self.assertEqual(len(resp.citations), 1)
        self.assertIn("30 calendar days", resp.text)
        self.mock_vector_store.retrieve.assert_called_once()
        self.mock_gemini_client.generate_response.assert_called_once()

    def test_explicit_order_id_calls_order_tool(self):
        """Explicit ORD-1007 triggers order lookup tool and order response generation."""
        self.mock_gemini_client.generate_order_response.return_value = LLMResponse(
            text="Your order ORD-1007 is in transit with UPS, tracking number 1ZAR100700000007, arriving August 22, 2026.",
            model_name="mock-model",
            analysis_status="order_lookup",
            handoff_required=False,
            evidence_count=0,
            evidence_chunk_ids=[],
            citations=[],
        )

        resp = self.orchestrator.process_turn("Where is ORD-1007?", session_id="s_ord1")
        self.assertEqual(resp.intent, "order")
        self.assertTrue(resp.tool_called)
        self.assertEqual(resp.tool_name, "order_lookup")
        self.assertEqual(resp.order_id_used, "ORD-1007")
        self.assertFalse(resp.handoff_required)
        self.mock_gemini_client.generate_order_response.assert_called_once()

    def test_lowercase_ord_id_normalization(self):
        """Lowercase 'ord-1007' normalizes to ORD-1007 and executes lookup."""
        self.mock_gemini_client.generate_order_response.return_value = LLMResponse(
            text="Order ord-1007 is shipped.",
            model_name="mock-model",
            analysis_status="order_lookup",
            handoff_required=False,
            evidence_count=0,
            evidence_chunk_ids=[],
            citations=[],
        )

        resp = self.orchestrator.process_turn("Track ord-1007", session_id="s_lower")
        self.assertEqual(resp.intent, "order")
        self.assertTrue(resp.tool_called)
        self.assertEqual(resp.order_id_used, "ORD-1007")

    def test_missing_order_id_does_not_call_order_tool(self):
        """'Where is my order?' without active session order ID prompts for ID without calling tool."""
        resp = self.orchestrator.process_turn("Where is my order?", session_id="s_missing")
        self.assertEqual(resp.intent, "order")
        self.assertFalse(resp.tool_called)
        self.assertIsNone(resp.order_id_used)
        self.assertIn("provide your order id", resp.text.lower())
        self.mock_gemini_client.generate_order_response.assert_not_called()

    def test_order_follow_up_uses_session_order_id(self):
        """Turn 2 'When will it arrive?' reuses stored ORD-1007 from Turn 1."""
        self.mock_gemini_client.generate_order_response.return_value = LLMResponse(
            text="It is estimated to arrive on August 22, 2026.",
            model_name="mock-model",
            analysis_status="order_lookup",
            handoff_required=False,
            evidence_count=0,
            evidence_chunk_ids=[],
            citations=[],
        )

        # Turn 1
        self.orchestrator.process_turn("Where is ORD-1007?", session_id="s_multi")

        # Turn 2
        resp2 = self.orchestrator.process_turn("When will it arrive?", session_id="s_multi")
        self.assertEqual(resp2.intent, "order")
        self.assertTrue(resp2.tool_called)
        self.assertEqual(resp2.order_id_used, "ORD-1007")

    def test_warranty_topic_switch_does_not_call_order_tool(self):
        """Switching from order to warranty policy calls RAG and does not execute order lookup."""
        self.mock_gemini_client.generate_order_response.return_value = LLMResponse(
            text="Order status", model_name="mock", analysis_status="order_lookup",
            handoff_required=False, evidence_count=0, evidence_chunk_ids=[], citations=[]
        )
        self.mock_vector_store.retrieve.return_value = [self.sample_chunk]
        self.mock_gemini_client.generate_response.return_value = LLMResponse(
            text="Aster & Row offers warranty coverage.", model_name="mock",
            analysis_status="consistent", handoff_required=False, evidence_count=1,
            evidence_chunk_ids=["doc_01_chunk_01"], citations=[]
        )

        # Turn 1: Order
        self.orchestrator.process_turn("Where is ORD-1007?", session_id="s_switch")

        # Turn 2: Warranty
        resp2 = self.orchestrator.process_turn("What is your warranty policy?", session_id="s_switch")
        self.assertEqual(resp2.intent, "knowledge")
        self.assertFalse(resp2.tool_called)
        self.mock_vector_store.retrieve.assert_called_once()

    def test_order_follow_up_after_intermediate_topic_switch(self):
        """Turn 3 'When will it arrive?' recovers stored ORD-1007 even after intermediate warranty turn."""
        self.mock_gemini_client.generate_order_response.return_value = LLMResponse(
            text="August 22, 2026", model_name="mock", analysis_status="order_lookup",
            handoff_required=False, evidence_count=0, evidence_chunk_ids=[], citations=[]
        )
        self.mock_vector_store.retrieve.return_value = [self.sample_chunk]
        self.mock_gemini_client.generate_response.return_value = LLMResponse(
            text="Warranty info", model_name="mock", analysis_status="consistent",
            handoff_required=False, evidence_count=1, evidence_chunk_ids=[], citations=[]
        )

        # Turn 1: Order
        self.orchestrator.process_turn("Where is ORD-1007?", session_id="s_turn3")
        # Turn 2: Warranty
        self.orchestrator.process_turn("What is your warranty policy?", session_id="s_turn3")
        # Turn 3: Order follow-up
        resp3 = self.orchestrator.process_turn("When will it arrive?", session_id="s_turn3")

        self.assertEqual(resp3.intent, "order")
        self.assertTrue(resp3.tool_called)
        self.assertEqual(resp3.order_id_used, "ORD-1007")

    def test_international_shipping_follow_up_context(self):
        """Turn 2 'What about Canada?' enriches contextual query with international shipping."""
        self.mock_vector_store.retrieve.return_value = [self.sample_chunk]
        self.mock_gemini_client.generate_response.return_value = LLMResponse(
            text="Shipping to Canada details.", model_name="mock", analysis_status="consistent",
            handoff_required=False, evidence_count=1, evidence_chunk_ids=[], citations=[]
        )

        # Turn 1
        self.orchestrator.process_turn("Do you ship internationally?", session_id="s_intl")
        # Turn 2
        resp2 = self.orchestrator.process_turn("What about Canada?", session_id="s_intl")

        self.assertEqual(resp2.intent, "knowledge")
        self.assertIn("international", self.mock_vector_store.retrieve.call_args[1]["query"].lower())

    def test_trailplus_return_follow_up_context(self):
        """Turn 2 'What about TrailPlus members?' enriches contextual query with return policy."""
        self.mock_vector_store.retrieve.return_value = [self.sample_chunk]
        self.mock_gemini_client.generate_response.return_value = LLMResponse(
            text="TrailPlus members get 45 days.", model_name="mock", analysis_status="consistent",
            handoff_required=False, evidence_count=1, evidence_chunk_ids=[], citations=[]
        )

        # Turn 1
        self.orchestrator.process_turn("What is the return policy?", session_id="s_ret")
        # Turn 2
        resp2 = self.orchestrator.process_turn("What about TrailPlus members?", session_id="s_ret")

        self.assertEqual(resp2.intent, "knowledge")
        self.assertIn("trailplus", self.mock_vector_store.retrieve.call_args[1]["query"].lower())

    def test_conflict_evidence_results_in_handoff(self):
        """Conflicting evidence triggers handoff_required=True."""
        chunk_a = SearchResult(
            chunk_id="doc_11_chunk", filename="11-product-care.md", document_title="Product Care",
            heading="Breeze Tumbler", content="The Breeze Tumbler is hand-wash only. Do not place in dishwasher.",
            metadata={"status": "active", "policy_authority": "official_policy", "audience": "customer"},
            policy_eligible=True, policy_reason="active", score=0.9, final_score=0.9
        )
        chunk_b = SearchResult(
            chunk_id="doc_12_chunk", filename="12-breeze-tumbler-product-card.md", document_title="Breeze Tumbler",
            heading="Cleaning", content="Dishwasher safe on the top rack.",
            metadata={"status": "active", "policy_authority": "official_product_card", "audience": "customer"},
            policy_eligible=True, policy_reason="active", score=0.9, final_score=0.9
        )
        self.mock_vector_store.retrieve.return_value = [chunk_a, chunk_b]
        self.mock_gemini_client.generate_response.return_value = LLMResponse(
            text="There is conflicting information regarding Breeze Tumbler dishwasher safety.",
            model_name="mock", analysis_status="conflict", handoff_required=True,
            evidence_count=2, evidence_chunk_ids=["doc_11_chunk", "doc_12_chunk"], citations=[]
        )

        resp = self.orchestrator.process_turn("Can I put the Breeze Tumbler in the dishwasher?", session_id="s_conf")
        self.assertEqual(resp.evidence_status, "conflict")
        self.assertTrue(resp.handoff_required)

    def test_insufficient_evidence_results_in_handoff(self):
        """Insufficient evidence triggers handoff_required=True."""
        weak_chunk = SearchResult(
            chunk_id="doc_01_chunk", filename="01-returns-policy-current.md", document_title="Returns",
            heading="General", content="Return policy rules.",
            metadata={"status": "active", "policy_authority": "official_policy", "audience": "customer"},
            policy_eligible=True, policy_reason="active", score=0.25, final_score=0.25
        )
        self.mock_vector_store.retrieve.return_value = [weak_chunk]
        self.mock_gemini_client.generate_response.return_value = LLMResponse(
            text="Official documentation does not contain information about vegan leather.",
            model_name="mock", analysis_status="insufficient", handoff_required=True,
            evidence_count=1, evidence_chunk_ids=["doc_01_chunk"], citations=[]
        )

        resp = self.orchestrator.process_turn("Are your backpacks made from vegan leather?", session_id="s_insuf")
        self.assertEqual(resp.evidence_status, "insufficient")
        self.assertTrue(resp.handoff_required)

    def test_order_exception_results_in_handoff(self):
        """ORD-1010 (status exception) triggers handoff_required=True."""
        self.mock_gemini_client.generate_order_response.return_value = LLMResponse(
            text="Shipment has an exception requiring support review.",
            model_name="mock", analysis_status="order_lookup", handoff_required=True,
            evidence_count=0, evidence_chunk_ids=[], citations=[]
        )

        resp = self.orchestrator.process_turn("Where is ORD-1010?", session_id="s_exc")
        self.assertEqual(resp.intent, "order")
        self.assertTrue(resp.tool_called)
        self.assertTrue(resp.handoff_required)

    def test_raw_internal_order_fields_never_enter_gemini_prompt(self):
        """Verify internal fields (risk_score, warehouse_note) are stripped before LLM call."""
        self.mock_gemini_client.generate_order_response.return_value = LLMResponse(
            text="Order status", model_name="mock", analysis_status="order_lookup",
            handoff_required=False, evidence_count=0, evidence_chunk_ids=[], citations=[]
        )

        self.orchestrator.process_turn("Where is ORD-1005?", session_id="s_sec")

        order_data_arg = self.mock_gemini_client.generate_order_response.call_args[1]["order_data"]
        payload_str = json.dumps(order_data_arg)
        self.assertNotIn("risk_score", payload_str)
        self.assertNotIn("warehouse_note", payload_str)
        self.assertNotIn("coupon", payload_str)
        self.assertNotIn("customer", order_data_arg)
        self.assertNotIn("shipping_address", payload_str)
        self.assertNotIn("email", payload_str)

    def test_raw_orders_json_never_included_in_prompt(self):
        """Verify prompt only contains single order dictionary, not all orders."""
        self.mock_gemini_client.generate_order_response.return_value = LLMResponse(
            text="Order status", model_name="mock", analysis_status="order_lookup",
            handoff_required=False, evidence_count=0, evidence_chunk_ids=[], citations=[]
        )

        self.orchestrator.process_turn("Where is ORD-1007?", session_id="s_single")

        order_data_arg = self.mock_gemini_client.generate_order_response.call_args[1]["order_data"]
        self.assertEqual(order_data_arg["order_id"], "ORD-1007")
        self.assertNotIn("ORD-1001", json.dumps(order_data_arg))

    def test_citations_for_knowledge_responses_validated(self):
        """Validated citations from LLM are preserved in AgentTurnResponse."""
        self.mock_vector_store.retrieve.return_value = [self.sample_chunk]
        self.mock_gemini_client.generate_response.return_value = LLMResponse(
            text="Answer text\n\nSources:\n- 01-returns-policy-current.md — Standard return window",
            model_name="mock", analysis_status="consistent", handoff_required=False,
            evidence_count=1, evidence_chunk_ids=["doc_01_chunk_01"],
            citations=["01-returns-policy-current.md — Standard return window"]
        )

        resp = self.orchestrator.process_turn("Return window query", session_id="s_cit")
        self.assertEqual(resp.citations, ["01-returns-policy-current.md — Standard return window"])

    def test_session_memory_remains_bounded(self):
        """Verify that multi-turn history never exceeds 5 turns in session context."""
        session = self.orchestrator.get_or_create_session("s_bound")
        for i in range(10):
            self.orchestrator.process_turn(f"Hello {i}", session_id="s_bound")

        self.assertLessEqual(len(session.recent_turns), 5)

    def test_two_separate_sessions_isolated(self):
        """Two sessions maintain completely distinct order IDs."""
        self.mock_gemini_client.generate_order_response.return_value = LLMResponse(
            text="Order info", model_name="mock", analysis_status="order_lookup",
            handoff_required=False, evidence_count=0, evidence_chunk_ids=[], citations=[]
        )

        self.orchestrator.process_turn("Where is ORD-1007?", session_id="user_1")
        self.orchestrator.process_turn("Where is ORD-1001?", session_id="user_2")

        sess1 = self.orchestrator.get_or_create_session("user_1")
        sess2 = self.orchestrator.get_or_create_session("user_2")

        self.assertEqual(sess1.last_order_id, "ORD-1007")
        self.assertEqual(sess2.last_order_id, "ORD-1001")

    def test_malformed_order_id_does_not_call_lookup(self):
        """Malformed order ID 'ORD-XYZ' returns clarification without calling lookup tool."""
        resp = self.orchestrator.process_turn("Where is ORD-XYZ?", session_id="s_mal")
        self.assertEqual(resp.intent, "order")
        self.assertFalse(resp.tool_called)
        self.assertIsNone(resp.order_id_used)
        self.assertIn("invalid", resp.text.lower())


if __name__ == "__main__":
    unittest.main()
