"""Heading-aware document chunking for Markdown policy documents."""

import re
from typing import Dict, List

from app.rag.models import Chunk, Document


def _slugify(text: str) -> str:
    """Convert heading text into a URL/anchor-friendly slug."""
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip().lower())
    return slug.strip("-") or "section"


def split_document(document: Document) -> List[Chunk]:
    """Split a Markdown document into chunks based on level-2 (##) headings.

    Behavior for pre-heading content:
    - If content exists before the first '##' heading (excluding the top-level '# Title'),
      it is extracted as a distinct chunk with heading='Overview'.
    - If no pre-heading content exists (only '# Title' and whitespace), no overview chunk is emitted.

    Args:
        document: The parsed Document to split.

    Returns:
        A list of Chunk objects preserving document metadata and section headings.
    """
    body = document.body.replace("\r\n", "\n")
    lines = body.split("\n")

    chunks: List[Chunk] = []
    current_heading: str = ""
    current_lines: List[str] = []
    seen_first_h2 = False
    pre_h2_lines: List[str] = []
    slug_counts: Dict[str, int] = {}

    def _make_chunk(heading: str, lines_to_join: List[str]) -> None:
        raw_content = "\n".join(lines_to_join).strip()
        if not raw_content:
            return

        base_slug = _slugify(heading)
        count = slug_counts.get(base_slug, 0)
        slug_counts[base_slug] = count + 1
        slug = base_slug if count == 0 else f"{base_slug}-{count + 1}"

        chunk_id = f"{document.filename}#{slug}"
        chunk_metadata = dict(document.metadata)
        chunk_metadata["heading"] = heading
        chunk_metadata["document_title"] = document.title
        chunk_metadata["filename"] = document.filename

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                filename=document.filename,
                document_title=document.title,
                heading=heading,
                content=raw_content,
                metadata=chunk_metadata,
            )
        )

    for line in lines:
        h2_match = re.match(r"^##\s+(.+)$", line)
        if h2_match:
            if not seen_first_h2:
                # Handle any preamble before the first ## heading
                seen_first_h2 = True
                # Filter out top-level '# Title' from preamble
                cleaned_pre = [
                    pl for pl in pre_h2_lines if not re.match(r"^#\s+", pl.strip())
                ]
                _make_chunk("Overview", cleaned_pre)
            else:
                # Emit the completed ## section
                _make_chunk(current_heading, current_lines)

            current_heading = h2_match.group(1).strip()
            current_lines = []
        else:
            if not seen_first_h2:
                pre_h2_lines.append(line)
            else:
                current_lines.append(line)

    # Flush the final section or the whole document if no ## existed
    if seen_first_h2:
        _make_chunk(current_heading, current_lines)
    else:
        cleaned_pre = [
            pl for pl in pre_h2_lines if not re.match(r"^#\s+", pl.strip())
        ]
        _make_chunk("Overview", cleaned_pre)

    return chunks


def split_documents(documents: List[Document]) -> List[Chunk]:
    """Split a collection of documents into chunks.

    Args:
        documents: List of Document objects.

    Returns:
        Flat list of all Chunk objects across all documents.
    """
    all_chunks: List[Chunk] = []
    for doc in documents:
        all_chunks.extend(split_document(doc))
    return all_chunks
