"""Diagnostic script to inspect deterministic evidence analysis, conflict detection, and uncertainty."""

import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.rag import VectorStore, analyze_evidence

DEMO_QUERIES = [
    "What is the standard return window?",
    "Can I put the Breeze Tumbler in the dishwasher?",
    "Are all fabrics and adhesives in your bags vegan?",
    "Do you ship internationally?",
]


def main() -> None:
    storage_dir = repo_root / "storage"
    if not (storage_dir / "kb_index.faiss").is_file():
        print(f"Error: Storage index not found at {storage_dir}", file=sys.stderr)
        sys.exit(1)

    store = VectorStore.load(storage_dir)

    print("=" * 90)
    print("ASTER & ROW EVIDENCE ANALYSIS & CONFLICT DETECTION DIAGNOSTIC")
    print("=" * 90)

    for i, q in enumerate(DEMO_QUERIES, 1):
        print(f"\n[QUERY {i}] \"{q}\"")
        print("-" * 90)
        results = store.retrieve(q, top_k=5, mode="customer")
        analysis = analyze_evidence(results, query=q)

        print(f"Status           : {analysis.status.upper()}")
        print(f"Handoff Required : {analysis.handoff_required}")
        print(f"Reason           : {analysis.reason}")
        print(f"Sources Inspected: {len(analysis.sources)} chunk(s)")

        if analysis.conflicting_claims:
            print("Detected Contradictions:")
            for c_idx, c in enumerate(analysis.conflicting_claims, 1):
                print(f"  ({c_idx}) Topic: {c.topic}")
                print(f"      Source A: {c.source_a} > {c.heading_a}")
                print(f"      Claim A : \"{c.claim_a}\"")
                print(f"      Source B: {c.source_b} > {c.heading_b}")
                print(f"      Claim B : \"{c.claim_b}\"")

        print(f"Structured Log   : {json.dumps(analysis.to_dict(), indent=2)}")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
