"""Deterministic intent classification and order ID extraction router."""

from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, Optional

from app.agent.session import SessionContext
from app.orders.service import normalize_order_id


@dataclass(frozen=True)
class RouteDecision:
    """Deterministic routing decision for a conversational turn."""

    intent: str  # "order" | "knowledge" | "greeting" | "unknown"
    order_id: Optional[str] = None
    normalized_order_id: Optional[str] = None
    topic: Optional[str] = None
    entity: Optional[str] = None
    reason: Optional[str] = None
    is_follow_up: bool = False
    contextual_query: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize route decision for observability."""
        return asdict(self)


class Router:
    """Deterministic rule-based router for support queries."""

    GREETING_PATTERN = re.compile(
        r"^\s*(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening)|hiya)(\s+(there|friend|everyone|team))?\b[!.?\s]*$",
        re.I,
    )

    # Matches order ID patterns with explicit separator (ORD-1007, ord#1007, ORD 1007) or direct 4 digits (ORD1007)
    # Strictly avoids matching common English words like 'order'
    ORDER_ID_PATTERN = re.compile(r"\b(?:ORD|ord)[\s#_\-]+[a-zA-Z0-9]+\b|\b(?:ORD|ord)\d{4}\b", re.I)

    ORDER_INTENT_KEYWORDS = re.compile(
        r"\b(where\s+is\s+my\s+(order|package|shipment)|track\s+my\s+order|status\s+of\s+my\s+order|"
        r"check\s+(my\s+)?(order|package|shipment)\s*status|order\s+status|"
        r"has\s+my\s+shipment\s+arrived|when\s+will\s+it\s+arrive|when\s+is\s+it\s+arriving|"
        r"where\s+is\s+it|is\s+it\s+on\s+the\s+way|is\s+my\s+order\s+delayed|tracking\s+number|"
        r"cancel\s+my\s+order|change\s+my\s+order\s+address)\b",
        re.I,
    )

    KNOWLEDGE_TOPICS: Dict[str, re.Pattern] = {
        "international_shipping": re.compile(
            r"\b(international|internationally|canada|canadian|germany|europe|uk|mexico|australia|customs|duties|foreign|overseas|ship\s+to|shipping\s+destinations|destination|countries|country)\b",
            re.I,
        ),
        "return_policy": re.compile(
            r"\b(return|returns|return\s+window|refund|refunds|exchange|exchanges|final\s+sale|30\s+days|45\s+days|60\s+days)\b",
            re.I,
        ),
        "membership": re.compile(
            r"\b(trailplus|trail\s+plus|membership|member\s+benefits)\b",
            re.I,
        ),
        "warranty": re.compile(
            r"\b(warranty|guarantee|defect|defective|broken\s+zipper|repair|repairs)\b",
            re.I,
        ),
        "product_care": re.compile(
            r"\b(dishwasher|wash|washing|clean|cleaning|care\s+instructions|hand-wash|hand\s+wash|tumbler|backpack\s+care)\b",
            re.I,
        ),
        "cancellations_and_changes": re.compile(
            r"\b(cancellation\s+policy|order\s+change\s+policy|address\s+change\s+policy|30\s+minutes)\b",
            re.I,
        ),
        "gift_cards_and_pricing": re.compile(
            r"\b(gift\s+card|gift\s+cards|price\s+adjustment|price\s+match)\b",
            re.I,
        ),
        "shipping_general": re.compile(
            r"\b(domestic\s+shipping|free\s+shipping|shipping\s+charges|shipping\s+fee|shipping\s+rates)\b",
            re.I,
        ),
        "products_and_materials": re.compile(
            r"\b(backpack|backpacks|bag|bags|tumbler|tumblers|materials|fabric|leather|vegan|zipper|sizing|specs)\b",
            re.I,
        ),
    }

    @classmethod
    def route(cls, query: str, session: Optional[SessionContext] = None) -> RouteDecision:
        """Deterministically route a customer message to an intent category.

        Args:
            query: The user message string.
            session: Optional conversational SessionContext.

        Returns:
            RouteDecision containing intent, extracted parameters, and reason.
        """
        trimmed = query.strip()
        if not trimmed:
            return RouteDecision(intent="unknown", reason="empty_query")

        # 1. Pure Greetings
        if cls.GREETING_PATTERN.match(trimmed):
            return RouteDecision(intent="greeting", reason="greeting_detected")

        # 2. Check for Explicit Order ID Pattern
        order_id_matches = cls.ORDER_ID_PATTERN.findall(trimmed)
        if order_id_matches:
            raw_match = order_id_matches[0]
            norm_id, err = normalize_order_id(raw_match)
            if norm_id is not None:
                return RouteDecision(
                    intent="order",
                    order_id=raw_match,
                    normalized_order_id=norm_id,
                    is_follow_up=False,
                    reason="explicit_order_id",
                )
            else:
                # Malformed order ID string (e.g. 'ORD-XYZ')
                return RouteDecision(
                    intent="order",
                    order_id=raw_match,
                    normalized_order_id=None,
                    is_follow_up=False,
                    reason="malformed_order_id",
                )

        # 3. Check for Order Follow-up / Order Inquiry without explicit ID
        if cls.ORDER_INTENT_KEYWORDS.search(trimmed):
            if session and session.last_order_id:
                return RouteDecision(
                    intent="order",
                    order_id=session.last_order_id,
                    normalized_order_id=session.last_order_id,
                    is_follow_up=True,
                    reason="session_order_follow_up",
                )
            return RouteDecision(
                intent="order",
                order_id=None,
                normalized_order_id=None,
                is_follow_up=False,
                reason="order_intent_missing_id",
            )

        # 4. Contextual Knowledge Follow-Ups
        query_lower = trimmed.lower()
        if session and session.last_topic:
            # Elliptic international shipping follow-up ("What about Canada?")
            if session.last_topic == "international_shipping" and re.search(r"\bcanada\b", query_lower):
                return RouteDecision(
                    intent="knowledge",
                    topic="international_shipping",
                    entity="Canada",
                    is_follow_up=True,
                    contextual_query=f"{trimmed} (Topic: international shipping destinations and fees)",
                    reason="international_shipping_follow_up",
                )

            # Elliptic return policy membership follow-up ("What about TrailPlus members?")
            if session.last_topic in ("return_policy", "returns") and re.search(r"\btrailplus\b", query_lower):
                return RouteDecision(
                    intent="knowledge",
                    topic="return_policy",
                    entity="TrailPlus",
                    is_follow_up=True,
                    contextual_query=f"{trimmed} (Topic: return policy for TrailPlus members)",
                    reason="return_policy_membership_follow_up",
                )

        # 5. Direct Knowledge Topics
        for topic_name, pattern in cls.KNOWLEDGE_TOPICS.items():
            if pattern.search(trimmed):
                return RouteDecision(
                    intent="knowledge",
                    topic=topic_name,
                    is_follow_up=False,
                    contextual_query=trimmed,
                    reason=f"knowledge_{topic_name}",
                )

        # 6. Fallback to Unknown / Ambiguous
        return RouteDecision(
            intent="unknown",
            reason="ambiguous_query",
        )


def route_query(query: str, session: Optional[SessionContext] = None) -> RouteDecision:
    """Functional convenience wrapper for Router.route."""
    return Router.route(query=query, session=session)
