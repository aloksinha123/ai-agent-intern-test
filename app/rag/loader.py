"""Document loader for parsing Markdown files with YAML front matter."""

from datetime import date, datetime
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple, Union

import yaml

from app.rag.models import Document


class MalformedFrontMatterError(ValueError):
    """Raised when a Markdown document has invalid, missing, or malformed YAML front matter."""


_FRONT_MATTER_PATTERN = re.compile(
    r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)$",
    re.DOTALL,
)


def _normalize_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively normalize metadata values, converting dates/datetimes to ISO strings."""
    normalized: Dict[str, Any] = {}
    for key, val in data.items():
        if isinstance(val, datetime):
            normalized[key] = val.isoformat()
        elif isinstance(val, date):
            normalized[key] = val.isoformat()
        elif isinstance(val, dict):
            normalized[key] = _normalize_metadata(val)
        elif isinstance(val, list):
            normalized[key] = [
                v.isoformat() if isinstance(v, (datetime, date)) else v for v in val
            ]
        else:
            normalized[key] = val
    return normalized


def parse_front_matter(raw_text: str, filename: str = "") -> Tuple[Dict[str, Any], str]:
    """Parse YAML front matter from a Markdown string.

    Args:
        raw_text: The complete raw text of the Markdown file.
        filename: Optional filename for detailed error reporting.

    Returns:
        A tuple of (metadata_dict, body_text).

    Raises:
        MalformedFrontMatterError: If front-matter delimiters are missing,
            YAML parsing fails, or front matter is not a key-value mapping.
    """
    file_context = f" in '{filename}'" if filename else ""

    if not raw_text.startswith("---"):
        raise MalformedFrontMatterError(
            f"Missing opening front-matter delimiter '---'{file_context}."
        )

    match = _FRONT_MATTER_PATTERN.match(raw_text)
    if not match:
        raise MalformedFrontMatterError(
            f"Missing closing front-matter delimiter '---'{file_context}."
        )

    yaml_block, body = match.group(1), match.group(2)

    try:
        parsed = yaml.safe_load(yaml_block)
    except yaml.YAMLError as exc:
        raise MalformedFrontMatterError(
            f"Failed to parse YAML front matter{file_context}: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise MalformedFrontMatterError(
            f"Front matter{file_context} must be a key-value mapping, got {type(parsed).__name__}."
        )

    normalized_metadata = _normalize_metadata(parsed)
    return normalized_metadata, body


def load_document(file_path: Union[str, Path]) -> Document:
    """Load a single Markdown document from disk and parse its front matter.

    Args:
        file_path: Path to the Markdown file.

    Returns:
        A populated Document object.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Knowledge-base file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    metadata, body = parse_front_matter(raw_text, filename=path.name)

    title = str(metadata.get("title", ""))
    if not title:
        # Fallback to first level-1 heading in body if title is not in metadata
        title_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else path.stem

    return Document(
        filename=path.name,
        filepath=str(path),
        title=title,
        metadata=metadata,
        body=body,
    )


def load_knowledge_base(dir_path: Union[str, Path]) -> List[Document]:
    """Load and parse all Markdown documents from a knowledge-base directory.

    Args:
        dir_path: Directory containing knowledge-base markdown files.

    Returns:
        A sorted list of Document objects.
    """
    path = Path(dir_path)
    if not path.is_dir():
        raise NotADirectoryError(f"Knowledge-base directory not found: {path}")

    md_files = sorted(path.glob("*.md"))
    return [load_document(f) for f in md_files]
