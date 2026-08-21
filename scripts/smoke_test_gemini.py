"""Live smoke test script for Gemini LLM orchestrator."""

from pathlib import Path
import sys

from dotenv import load_dotenv

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

load_dotenv(repo_root / ".env")

from app.llm import GeminiClient
from app.rag import VectorStore, analyze_evidence


def main() -> None:
    storage_dir = repo_root / "storage"
    if not (storage_dir / "kb_index.faiss").is_file():
        print("Error: Vector store index not found.", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("LIVE GEMINI ORCHESTRATION SMOKE TEST")
    print("=" * 80)

    try:
        store = VectorStore.load(storage_dir)
        client = GeminiClient()

        query = "What is the standard return window?"
        print(f"\nUser Query: {query}")

        evidence = store.retrieve(query, top_k=5, mode="customer")
        analysis = analyze_evidence(evidence, query=query)

        print(f"Evidence Analysis Status: {analysis.status}")
        print(f"Handoff Required        : {analysis.handoff_required}")

        response = client.generate_response(query=query, evidence=evidence, analysis=analysis)

        print("\n--- Model Generated Response ---")
        print(response.text)
        print("--------------------------------")
        print(f"Model Name           : {response.model_name}")
        print(f"Validated Citations  : {response.citations}")
        print(f"Handoff Flag (Final) : {response.handoff_required}")
        print("\nSmoke test succeeded.")

    except Exception as e:
        print(f"\nLive smoke test encountered error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
