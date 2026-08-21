"""Standalone order lookup tool function for agent orchestration."""

from pathlib import Path
from typing import Optional, Union

from app.orders.models import OrderLookupResult
from app.orders.service import OrderService

_DEFAULT_SERVICE: Optional[OrderService] = None


def get_order_service(data_path: Optional[Union[str, Path]] = None) -> OrderService:
    """Get or create singleton OrderService instance."""
    global _DEFAULT_SERVICE
    if data_path is not None:
        return OrderService(data_path=data_path)
    if _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = OrderService()
    return _DEFAULT_SERVICE


def lookup_order(
    order_id: Optional[str],
    data_path: Optional[Union[str, Path]] = None,
) -> OrderLookupResult:
    """Look up order information and return a sanitized, customer-safe result.

    Args:
        order_id: User-supplied order ID (e.g. 'ORD-1001', 'ord-1007', 'ORD 1007').
        data_path: Optional custom path to orders.json file.

    Returns:
        OrderLookupResult with customer-safe fields or structured error code.
    """
    service = get_order_service(data_path=data_path)
    return service.lookup(raw_id=order_id)
