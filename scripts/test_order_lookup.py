"""Diagnostic script to safely test deterministic order lookups and inspect sanitized outputs."""

import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.orders import lookup_order

TEST_ORDER_IDS = [
    "ORD-1007",  # Shipped with carrier and ETA
    "ORD-1004",  # Cancelled (suppressed tracking & ETA)
    "ORD-1011",  # Shipped with null ETA feed
    "ORD-1010",  # Exception status (requires handoff)
    "ORD-9999",  # Unknown ID (order_not_found, requires handoff)
]


def main() -> None:
    print("=" * 80)
    print("ASTER & ROW DETERMINISTIC ORDER LOOKUP DIAGNOSTIC")
    print("=" * 80)

    for oid in TEST_ORDER_IDS:
        print(f"\n[LOOKUP] Input: \"{oid}\"")
        print("-" * 80)

        result = lookup_order(oid)
        print(f"Found               : {result.found}")
        print(f"Normalized Order ID : {result.normalized_order_id}")
        print(f"Handoff Required    : {result.handoff_required}")
        print(f"Error Code          : {result.error_code}")
        print(f"Error Message       : {result.error_message}")

        if result.order:
            order = result.order
            print(f"Status              : {order.status.upper()}")
            print(f"Membership Tier     : {order.membership_tier}")
            print(f"Carrier             : {order.carrier}")
            print(f"Tracking Number     : {order.tracking_number}")
            print(f"Estimated Delivery  : {order.estimated_delivery}")
            print(f"Safe Message        : {order.customer_safe_message}")
            items_str = ", ".join([f"{it.name} (x{it.quantity})" for it in order.items])
            print(f"Items               : {items_str}")

        print("\nSanitized JSON Payload:")
        print(json.dumps(result.to_dict(), indent=2))

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
