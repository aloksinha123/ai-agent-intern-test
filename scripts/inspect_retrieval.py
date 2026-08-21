"""Diagnostic script to inspect metadata-aware retrieval results and policy decisions."""

from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.rag import VectorStore

AUDIT_QUERIES = [
    "What is the standard return window?",
    "How long do TrailPlus members have to return an item?",
    "Do you ship internationally?",
    "Can I put the Breeze Tumbler in the dishwasher?",
    "Do you offer a 60 day return policy?",
]


def main() -> None:
    storage_dir = repo_root / "storage"
    if not (storage_dir / "kb_index.faiss").is_file():
        print(f"Error: Storage directory or index not found at {storage_dir}", file=sys.stderr)
        print("Run scripts/build_index.py first.", file=sys.stderr)
        sys.exit(1)

    store = VectorStore.load(storage_dir)

    print("=" * 90)
    print("METADATA-AWARE RETRIEVAL DIAGNOSTIC INSPECTION (CUSTOMER MODE)")
    print("=" * 90)

    for i, q in enumerate(AUDIT_QUERIES, 1):
        print(f"\n[QUERY {i}] \"{q}\"")
        print("-" * 90)
        print(f"{'Rank':<5} {'Score':<8} {'Final':<8} {'Policy Status':<26} {'Filename':<32} {'Heading'}")
        print("-" * 90)

        results = store.retrieve(q, top_k=5, mode="customer")
        for rank, res in enumerate(results, 1):
            print(
                f"{rank:<5} {res.score:<8.4f} {res.final_score:<8.4f} {res.policy_reason:<26} {res.filename:<32} {res.heading}"
            )
            preview = res.content.replace("\n", " ")[:100]
            print(f"      Preview: {preview}...")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
