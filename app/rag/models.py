"""Data models for knowledge-base documents and chunks."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class Document:
    """Represents an ingested knowledge-base Markdown document with metadata."""

    filename: str
    filepath: str
    title: str
    metadata: Dict[str, Any]
    body: str

    @property
    def document_id(self) -> str:
        """Helper to get document_id from metadata if present."""
        return str(self.metadata.get("document_id", ""))

    @property
    def status(self) -> str:
        """Helper to get status from metadata if present."""
        return str(self.metadata.get("status", ""))

    @property
    def policy_authority(self) -> str:
        """Helper to get policy_authority from metadata if present."""
        return str(self.metadata.get("policy_authority", ""))

    @property
    def audience(self) -> str:
        """Helper to get audience from metadata if present."""
        return str(self.metadata.get("audience", ""))


@dataclass(frozen=True)
class Chunk:
    """Represents a discrete heading-level section chunk from a document."""

    chunk_id: str
    filename: str
    document_title: str
    heading: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """Represents a scored retrieval result from the vector index with policy metadata."""

    score: float
    chunk_id: str
    filename: str
    document_title: str
    heading: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    policy_eligible: bool = True
    policy_reason: str = "eligible"
    final_score: float = 0.0

