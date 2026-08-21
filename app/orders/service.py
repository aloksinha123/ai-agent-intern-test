"""Deterministic order data loading, ID normalization, and customer-safe sanitization."""

import json
from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple, Union

from app.orders.models import CustomerOrderView, OrderItem, OrderLookupResult

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "orders.json"


def normalize_order_id(raw_id: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Normalize user input to canonical uppercase 'ORD-XXXX' format.

    Handles harmless variations:
    - Lowercase: 'ord-1001' -> 'ORD-1001'
    - Whitespace & surrounding punctuation: '  ORD-1001.  ' -> 'ORD-1001'
    - Separator variations: 'ORD 1001', 'ord#1001', 'ORD_1001' -> 'ORD-1001'

    Returns:
        Tuple of (normalized_id, error_code).
        If valid: (canonical_id, None)
        If missing: (None, 'missing_order_id')
        If malformed: (None, 'invalid_order_id')
    """
    if raw_id is None:
        return None, "missing_order_id"

    trimmed = str(raw_id).strip()
    if not trimmed:
        return None, "missing_order_id"

    # Strip harmless surrounding punctuation (quotes, trailing dots, commas, parens)
    cleaned = trimmed.strip(".,:;!?'\"()[]{}#")
    cleaned = cleaned.upper()

    # Normalize obvious separator variations between 'ORD' and the 4 digits
    cleaned = re.sub(r"^ORD[\s#_\-]+(\d{4})$", r"ORD-\1", cleaned)

    # Validate strict canonical format: ^ORD-\d{4}$
    if re.match(r"^ORD-\d{4}$", cleaned):
        return cleaned, None

    return None, "invalid_order_id"


class OrderService:
    """Read-only service managing order data loading and customer-safe lookups."""

    def __init__(self, data_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize service and load orders dataset into memory.

        Args:
            data_path: Optional path to orders.json file.
        """
        self.data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH
        self.snapshot_at: str = ""
        self.orders_by_id: Dict[str, Dict[str, Any]] = {}
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load and index orders from JSON file."""
        if not self.data_path.is_file():
            raise FileNotFoundError(f"Orders dataset not found at: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.snapshot_at = data.get("snapshot_at", "")
        self.orders_by_id = {
            order["order_id"]: order for order in data.get("orders", [])
        }

    def lookup(self, raw_id: Optional[str]) -> OrderLookupResult:
        """Look up an order deterministically with customer-safe sanitization.

        Args:
            raw_id: Raw order ID string from user or caller.

        Returns:
            OrderLookupResult containing sanitized customer view or error.
        """
        normalized_id, error_code = normalize_order_id(raw_id)

        if error_code == "missing_order_id":
            return OrderLookupResult(
                found=False,
                normalized_order_id=None,
                order=None,
                error_code="missing_order_id",
                error_message="No order ID provided",
                handoff_required=False,
            )

        if error_code == "invalid_order_id":
            return OrderLookupResult(
                found=False,
                normalized_order_id=None,
                order=None,
                error_code="invalid_order_id",
                error_message=f"Invalid order ID format: '{raw_id}'",
                handoff_required=False,
            )

        if normalized_id not in self.orders_by_id:
            return OrderLookupResult(
                found=False,
                normalized_order_id=normalized_id,
                order=None,
                error_code="order_not_found",
                error_message=f"Order '{normalized_id}' not found in records",
                handoff_required=True,
            )

        raw = self.orders_by_id[normalized_id]
        status = str(raw.get("status", "")).lower()

        # Build customer-safe items list (strictly excluding internal SKUs from customer view)
        safe_items = [
            OrderItem(
                name=item["name"],
                quantity=int(item["quantity"]),
                final_sale=bool(item.get("final_sale", False)),
            )
            for item in raw.get("items", [])
        ]

        carrier = raw.get("carrier")
        tracking = raw.get("tracking_number")
        estimated_delivery = raw.get("estimated_delivery")
        handoff_required = False

        # Status-based sanitization rules
        if status == "cancelled":
            # Suppress stale label / tracking / ETA records created prior to cancellation
            carrier = None
            tracking = None
            estimated_delivery = None
        elif status == "returned":
            # Suppress obsolete delivery ETA
            estimated_delivery = None
        elif status == "exception":
            # Carrier exception requires human support review
            estimated_delivery = None
            handoff_required = True
        elif status == "shipped" and estimated_delivery is None:
            # Keep ETA as None (do not invent dates)
            estimated_delivery = None

        customer_view = CustomerOrderView(
            order_id=raw["order_id"],
            membership_tier=raw.get("membership_tier", "standard"),
            items=safe_items,
            placed_at=raw["placed_at"],
            status=raw["status"],
            status_updated_at=raw["status_updated_at"],
            shipped_at=raw.get("shipped_at"),
            delivered_at=raw.get("delivered_at"),
            carrier=carrier,
            tracking_number=tracking,
            estimated_delivery=estimated_delivery,
            customer_safe_message=raw.get("customer_safe_message", ""),
        )

        return OrderLookupResult(
            found=True,
            normalized_order_id=normalized_id,
            order=customer_view,
            error_code=None,
            error_message=None,
            handoff_required=handoff_required,
        )
