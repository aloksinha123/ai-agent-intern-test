"""Debug script demonstrating privacy-safe structured tracing across scenarios."""

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.agent.orchestrator import AgentOrchestrator
from app.orders.service import OrderService
from app.rag.vector_store import VectorStore
from evaluation.run_evaluation import MockGeminiEvaluatorClient


def main() -> None:
    print("=" * 80)
    print("ASTER & ROW AGENT OBSERVABILITY & STRUCTURED TRACE DEMO")
    print("=" * 80)

    store = VectorStore.load()
    client = MockGeminiEvaluatorClient()
    service = OrderService()

    orchestrator = AgentOrchestrator(
        vector_store=store,
        gemini_client=client,
        order_service=service,
        enable_tracing=True,
    )

    scenarios = [
        ("Scenario 1: Standard Return Policy", "How long do I have to return an unused backpack?"),
        ("Scenario 2: Valid Order Status Lookup", "Where is my order ORD-1001?"),
        ("Scenario 3: Missing Order ID Prompt", "Where is my package?"),
        ("Scenario 4: Breeze Tumbler Dishwasher Conflict", "Can I put the Breeze Tumbler in the dishwasher?"),
        ("Scenario 5: Insufficient Information Abstention", "Are all fabrics and adhesives in your bags vegan?"),
    ]

    for title, query in scenarios:
        print("\n" + "#" * 80)
        print(f"### {title}")
        print(f"Query: \"{query}\"")
        print("#" * 80)

        response = orchestrator.process_turn(message=query, session_id=f"demo_{title.split()[1]}")

        if response.trace:
            trace_dict = response.trace.to_dict()
            print(json.dumps(trace_dict, indent=2))
        else:
            print("[ERROR] Trace was not generated.")

    print("\n" + "=" * 80)
    print("All scenarios executed with complete privacy-safe observability traces.")
    print("=" * 80)


if __name__ == "__main__":
    main()
