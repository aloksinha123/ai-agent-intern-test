"""Smoke test script verifying the 5 required conversational flows through AgentOrchestrator."""

import json
from pathlib import Path
import sys
from unittest.mock import MagicMock

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.agent.orchestrator import AgentOrchestrator
from app.llm.client import GeminiClient, LLMResponse
from app.orders.service import OrderService
from app.rag.vector_store import VectorStore


def create_mock_gemini_client() -> GeminiClient:
    """Create a mock Gemini client with realistic grounded responses."""
    client = MagicMock(spec=GeminiClient)

    def mock_generate_response(query, evidence, analysis):
        if analysis.status == "conflict":
            conflict_sources = []
            for c in analysis.conflicting_claims:
                s_a = f"{c.source_a} — {c.heading_a}"
                s_b = f"{c.source_b} — {c.heading_b}"
                if s_a not in conflict_sources:
                    conflict_sources.append(s_a)
                if s_b not in conflict_sources:
                    conflict_sources.append(s_b)

            sources_text = "\n".join([f"- {s}" for s in conflict_sources])
            return LLMResponse(
                text=f"Official documentation contains conflicting cleaning instructions for the Breeze Tumbler. Product care states hand-wash only, while the product card states dishwasher safe. We recommend contacting support.\n\nSources:\n{sources_text}",
                model_name="mock-gemini",
                analysis_status="conflict",
                handoff_required=True,
                evidence_count=len(evidence),
                evidence_chunk_ids=[s.chunk_id for s in evidence],
                citations=conflict_sources,
            )
        elif analysis.status == "insufficient":
            return LLMResponse(
                text="Official documentation does not contain the requested information.",
                model_name="mock-gemini",
                analysis_status="insufficient",
                handoff_required=True,
                evidence_count=len(evidence),
                evidence_chunk_ids=[s.chunk_id for s in evidence],
                citations=[],
            )
        else:
            return LLMResponse(
                text="The standard return window is 30 calendar days from delivery.\n\nSources:\n- 01-returns-policy-current.md — Standard return window",
                model_name="mock-gemini",
                analysis_status="consistent",
                handoff_required=False,
                evidence_count=len(evidence),
                evidence_chunk_ids=[s.chunk_id for s in evidence],
                citations=["01-returns-policy-current.md — Standard return window"],
            )

    def mock_generate_order_response(query, order_data, handoff_required):
        oid = order_data.get("order_id", "Unknown")
        status = order_data.get("status", "Unknown")
        msg = order_data.get("customer_safe_message", "")
        return LLMResponse(
            text=f"Your order {oid} is currently {status.upper()}. {msg}",
            model_name="mock-gemini",
            analysis_status="order_lookup",
            handoff_required=bool(handoff_required),
            evidence_count=0,
            evidence_chunk_ids=[],
            citations=[],
        )

    client.generate_response.side_effect = mock_generate_response
    client.generate_order_response.side_effect = mock_generate_order_response
    return client


def main() -> None:
    print("=" * 80)
    print("ASTER & ROW AGENT ORCHESTRATION INTEGRATION SMOKE TEST")
    print("=" * 80)

    # Initialize real vector store and real order service with mock Gemini
    vector_store = VectorStore.load()
    order_service = OrderService()
    gemini_client = create_mock_gemini_client()

    orchestrator = AgentOrchestrator(
        vector_store=vector_store,
        gemini_client=gemini_client,
        order_service=order_service,
    )

    flows = [
        ("Flow 1: Standard Return Window", [("sess_flow_1", "What is the standard return window?")]),
        ("Flow 2: Explicit Order Lookup", [("sess_flow_2", "Where is ORD-1007?")]),
        ("Flow 3: Order Intent Without ID", [("sess_flow_3", "Where is my order?")]),
        (
            "Flow 4: Multi-Turn Order Follow-Up",
            [
                ("sess_flow_4", "Where is ORD-1007?"),
                ("sess_flow_4", "When will it arrive?"),
            ],
        ),
        ("Flow 5: Conflicting Policy Query", [("sess_flow_5", "Can I put the Breeze Tumbler in the dishwasher?")]),
    ]

    for title, turns in flows:
        print(f"\n{title}")
        print("-" * 80)
        for session_id, message in turns:
            print(f"\n[USER -> {session_id}]: \"{message}\"")
            resp = orchestrator.process_turn(message, session_id=session_id)
            print(f"Intent            : {resp.intent}")
            print(f"Tool Called       : {resp.tool_called} (name: {resp.tool_name}, id: {resp.order_id_used})")
            print(f"Evidence Status   : {resp.evidence_status}")
            print(f"Handoff Required  : {resp.handoff_required}")
            print(f"Citations         : {resp.citations}")
            print(f"Assistant Response:\n{resp.text}")

    print("\n" + "=" * 80)
    print("All 5 smoke test flows completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
