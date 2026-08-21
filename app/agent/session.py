"""Bounded multi-turn conversational session context."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Turn:
    """Represents a single conversational turn."""

    role: str  # "user" | "assistant"
    text: str


@dataclass
class SessionContext:
    """Bounded conversational context preserving minimal required state.

    Invariants:
    - Bounded memory: keeps at most max_turns (default: 5) recent turns.
    - Preserves last order ID and topic without cross-session contamination.
    - Never stores raw tool payloads, PII, or full retrieved chunks.
    """

    session_id: str
    last_order_id: Optional[str] = None
    last_topic: Optional[str] = None
    current_entity: Optional[str] = None
    recent_turns: List[Turn] = field(default_factory=list)
    max_turns: int = 5

    def add_turn(self, role: str, text: str) -> None:
        """Add a turn and ensure history stays strictly bounded."""
        self.recent_turns.append(Turn(role=role, text=text.strip()))
        if len(self.recent_turns) > self.max_turns:
            self.recent_turns = self.recent_turns[-self.max_turns :]

    def set_order_context(self, order_id: str) -> None:
        """Update last active order context."""
        self.last_order_id = order_id
        self.last_topic = "order"

    def set_topic(self, topic: str, entity: Optional[str] = None) -> None:
        """Update last discussed policy or product topic."""
        self.last_topic = topic
        if entity is not None:
            self.current_entity = entity

    def clear_order_context(self) -> None:
        """Clear active order context."""
        self.last_order_id = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session state for debugging/observability."""
        return {
            "session_id": self.session_id,
            "last_order_id": self.last_order_id,
            "last_topic": self.last_topic,
            "current_entity": self.current_entity,
            "turn_count": len(self.recent_turns),
            "recent_turns": [asdict(t) for t in self.recent_turns],
        }
