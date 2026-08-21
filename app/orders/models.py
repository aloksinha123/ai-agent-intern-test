"""Data models for customer-safe order lookup results."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class OrderItem:
    """Customer-safe view of an item within an order."""

    name: str
    quantity: int
    final_sale: bool


@dataclass(frozen=True)
class CustomerOrderView:
    """Sanitized, customer-safe view of an order.

    Strictly excludes customer PII (name, email, shipping address)
    and internal metadata (risk scores, warehouse notes, triage tags).
    """

    order_id: str
    membership_tier: str
    items: List[OrderItem]
    placed_at: str
    status: str
    status_updated_at: str
    shipped_at: Optional[str]
    delivered_at: Optional[str]
    carrier: Optional[str]
    tracking_number: Optional[str]
    estimated_delivery: Optional[str]
    customer_safe_message: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize customer-safe order view to dictionary."""
        return {
            "order_id": self.order_id,
            "membership_tier": self.membership_tier,
            "items": [asdict(item) for item in self.items],
            "placed_at": self.placed_at,
            "status": self.status,
            "status_updated_at": self.status_updated_at,
            "shipped_at": self.shipped_at,
            "delivered_at": self.delivered_at,
            "carrier": self.carrier,
            "tracking_number": self.tracking_number,
            "estimated_delivery": self.estimated_delivery,
            "customer_safe_message": self.customer_safe_message,
        }


@dataclass(frozen=True)
class OrderLookupResult:
    """Result envelope for deterministic order lookup."""

    found: bool
    normalized_order_id: Optional[str] = None
    order: Optional[CustomerOrderView] = None
    error_code: Optional[str] = None  # "missing_order_id" | "invalid_order_id" | "order_not_found"
    error_message: Optional[str] = None
    handoff_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result envelope to safe dictionary."""
        return {
            "found": self.found,
            "normalized_order_id": self.normalized_order_id,
            "order": self.order.to_dict() if self.order else None,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "handoff_required": self.handoff_required,
        }
