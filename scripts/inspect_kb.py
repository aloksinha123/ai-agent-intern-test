"""Inspection script to load and chunk the knowledge base and print summary statistics."""

import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.rag import load_knowledge_base, split_document, split_documents


def main() -> None:
    kb_dir = repo_root / "knowledge-base"
    if not kb_dir.is_dir():
        print(f"Error: knowledge-base directory not found at {kb_dir}", file=sys.stderr)
        sys.exit(1)

    docs = load_knowledge_base(kb_dir)
    chunks = split_documents(docs)

    print("=" * 80)
    print("ASTER & ROW KNOWLEDGE BASE INGESTION & CHUNKING SUMMARY")
    print("=" * 80)
    print(f"Total Documents Ingested : {len(docs)}")
    print(f"Total Chunks Generated   : {len(chunks)}")
    print("-" * 80)
    print(f"{'Filename':<42} {'Doc ID':<15} {'Status':<12} {'Authority':<10} {'Chunks':<6}")
    print("-" * 80)

    for doc in docs:
        doc_chunks = split_document(doc)
        doc_id = str(doc.metadata.get("document_id", "N/A"))
        status = str(doc.metadata.get("status", "N/A"))
        authority = str(doc.metadata.get("policy_authority", "N/A"))
        print(f"{doc.filename:<42} {doc_id:<15} {status:<12} {authority:<10} {len(doc_chunks):<6}")

    print("=" * 80)
    print("SAMPLE CHUNK METADATA RECORDS")
    print("=" * 80)

    # Pick a few representative chunks to display
    sample_indices = [0, 4, 15, 30, len(chunks) - 1]
    for idx in sample_indices:
        if idx < len(chunks):
            c = chunks[idx]
            print(f"\n[Chunk {idx + 1}/{len(chunks)}] Chunk ID: {c.chunk_id}")
            print(f"  Heading        : {c.heading}")
            print(f"  Document Title : {c.document_title}")
            print(f"  Content Preview: {c.content[:90]}..." if len(c.content) > 90 else f"  Content Preview: {c.content}")
            print(f"  Metadata       : {json.dumps(c.metadata, default=str)}")


if __name__ == "__main__":
    main()
