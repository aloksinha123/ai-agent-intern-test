"""Script to build and persist the FAISS vector index using local Sentence Transformers."""

from pathlib import Path
import sys
import time

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.rag import (
    DEFAULT_EMBEDDING_MODEL,
    VectorStore,
    embed_texts,
    get_embedding_model_name,
    load_knowledge_base,
    split_documents,
)


def main() -> None:
    kb_dir = repo_root / "knowledge-base"
    storage_dir = repo_root / "storage"

    if not kb_dir.is_dir():
        print(f"Error: knowledge-base directory not found at {kb_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("ASTER & ROW LOCAL VECTOR INDEX BUILDER")
    print("=" * 80)

    # 1. Load documents and generate chunks
    docs = load_knowledge_base(kb_dir)
    chunks = split_documents(docs)
    print(f"Loaded {len(docs)} documents and produced {len(chunks)} chunks.")

    # 2. Prepare texts for embedding
    # Include document title and heading for grounded semantic context
    texts_to_embed = [
        f"{chunk.document_title} > {chunk.heading}\n\n{chunk.content}"
        for chunk in chunks
    ]

    model_name = get_embedding_model_name()
    print(f"Generating embeddings using local model: '{model_name}'...")

    start_time = time.time()
    try:
        embeddings = embed_texts(texts_to_embed, model_name=model_name)
    except Exception as exc:
        print(f"\n[ERROR] Local embedding generation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - start_time
    dim = embeddings.shape[1]
    print(f"Generated {len(embeddings)} embeddings in {elapsed:.2f}s (vector dimension: {dim}).")

    # 3. Build FAISS vector store
    print("Building FAISS IndexFlatIP (cosine similarity)...")
    store = VectorStore.build(chunks=chunks, embeddings=embeddings)

    # 4. Save to storage
    storage_dir.mkdir(parents=True, exist_ok=True)
    store.save(storage_dir)

    index_file = storage_dir / "kb_index.faiss"
    chunks_file = storage_dir / "kb_chunks.json"

    print("-" * 80)
    print("INDEX BUILD SUMMARY")
    print("-" * 80)
    print(f"Embedding Model      : {model_name}")
    print(f"Total Chunks Indexed : {len(chunks)}")
    print(f"Vector Dimension     : {dim}")
    print(f"FAISS Index File     : {index_file} ({index_file.stat().st_size} bytes)")
    print(f"Metadata Mapping File: {chunks_file} ({chunks_file.stat().st_size} bytes)")
    print("=" * 80)
    print("Vector index build complete and persisted successfully.")


if __name__ == "__main__":
    main()
