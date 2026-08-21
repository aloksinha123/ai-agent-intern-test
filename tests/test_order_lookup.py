"""Unit tests for deterministic order lookup, ID normalization, and security sanitization."""

import hashlib
import json
from pathlib import Path
import unittest

from app.orders.models import CustomerOrderView, OrderItem, OrderLookupResult
from app.orders.service import normalize_order_id
from app.orders.tool import get_order_service, lookup_order

REPO_ROOT = Path(__file__).resolve().parent.parent
ORDERS_JSON_PATH = REPO_ROOT / "data" / "orders.json"


class TestOrderIDNormalization(unittest.TestCase):
    """Test deterministic order ID normalization and validation."""

    def test_canonical_id_preserved(self):
        norm_id, err = normalize_order_id("ORD-1001")
        self.assertEqual(norm_id, "ORD-1001")
        self.assertIsNone(err)

    def test_lowercase_normalization(self):
        norm_id, err = normalize_order_id("ord-1007")
        self.assertEqual(norm_id, "ORD-1007")
        self.assertIsNone(err)

    def test_whitespace_and_punctuation_normalization(self):
        norm_id, err = normalize_order_id("  ORD-1007.  ")
        self.assertEqual(norm_id, "ORD-1007")
        self.assertIsNone(err)

    def test_alternate_separators_normalization(self):
        self.assertEqual(normalize_order_id("ORD 1007")[0], "ORD-1007")
        self.assertEqual(normalize_order_id("ord#1007")[0], "ORD-1007")
        self.assertEqual(normalize_order_id("ORD_1007")[0], "ORD-1007")

    def test_missing_id(self):
        for empty_val in [None, "", "   "]:
            norm_id, err = normalize_order_id(empty_val)
            self.assertIsNone(norm_id)
            self.assertEqual(err, "missing_order_id")

    def test_malformed_ids(self):
        for malformed in ["HELLO", "1234", "ORD-XYZ", "ORD-100", "ORD-10001", "ORD#ABC"]:
            norm_id, err = normalize_order_id(malformed)
            self.assertIsNone(norm_id)
            self.assertEqual(err, "invalid_order_id")

    def test_no_fuzzy_matching(self):
        """'ORD-101' or 'ORD-100' must not guess or map to ORD-1001 or ORD-1010."""
        norm_id, err = normalize_order_id("ORD-101")
        self.assertIsNone(norm_id)
        self.assertEqual(err, "invalid_order_id")


class TestOrderLookupTool(unittest.TestCase):
    """Test order lookup tool behavior, field allowlist, and status sanitization."""

    def test_valid_shipped_order_ord1007(self):
        """ORD-1007: shipped international order with valid tracking and ETA."""
        res = lookup_order("ORD-1007")
        self.assertTrue(res.found)
        self.assertEqual(res.normalized_order_id, "ORD-1007")
        self.assertIsNone(res.error_code)
        self.assertFalse(res.handoff_required)

        order = res.order
        self.assertIsNotNone(order)
        self.assertEqual(order.order_id, "ORD-1007")
        self.assertEqual(order.membership_tier, "standard")
        self.assertEqual(order.status, "shipped")
        self.assertEqual(order.carrier, "UPS")
        self.assertEqual(order.tracking_number, "1ZAR100700000007")
        self.assertEqual(order.estimated_delivery, "2026-08-22")
        self.assertEqual(len(order.items), 1)
        self.assertEqual(order.items[0].name, "Atlas Weekender")
        self.assertEqual(order.items[0].quantity, 1)

    def test_unknown_valid_id_ord9999(self):
        """ORD-9999: well-formed ID not in dataset returns order_not_found and handoff."""
        res = lookup_order("ORD-9999")
        self.assertFalse(res.found)
        self.assertEqual(res.normalized_order_id, "ORD-9999")
        self.assertEqual(res.error_code, "order_not_found")
        self.assertIn("not found", res.error_message.lower())
        self.assertTrue(res.handoff_required)
        self.assertIsNone(res.order)

    def test_missing_id_lookup(self):
        """Empty input returns missing_order_id."""
        res = lookup_order("")
        self.assertFalse(res.found)
        self.assertEqual(res.error_code, "missing_order_id")
        self.assertIsNone(res.normalized_order_id)
        self.assertFalse(res.handoff_required)

    def test_malformed_id_lookup(self):
        """Malformed input returns invalid_order_id."""
        res = lookup_order("INVALID_ID_123")
        self.assertFalse(res.found)
        self.assertEqual(res.error_code, "invalid_order_id")
        self.assertFalse(res.handoff_required)

    def test_cancelled_order_sanitization_ord1004(self):
        """ORD-1004: cancelled order must suppress stale carrier, tracking, and ETA."""
        res = lookup_order("ORD-1004")
        self.assertTrue(res.found)
        order = res.order
        self.assertEqual(order.status, "cancelled")
        self.assertIsNone(order.carrier)
        self.assertIsNone(order.tracking_number)
        self.assertIsNone(order.estimated_delivery)
        self.assertIn("cancelled", order.customer_safe_message.lower())

    def test_returned_order_sanitization_ord1008(self):
        """ORD-1008: returned order must suppress obsolete delivery ETA."""
        res = lookup_order("ORD-1008")
        self.assertTrue(res.found)
        order = res.order
        self.assertEqual(order.status, "returned")
        self.assertIsNone(order.estimated_delivery)
        self.assertIn("return was received", order.customer_safe_message.lower())

    def test_shipped_order_with_null_eta_ord1011(self):
        """ORD-1011: shipped order with unavailable ETA must keep ETA null without inventing date."""
        res = lookup_order("ORD-1011")
        self.assertTrue(res.found)
        order = res.order
        self.assertEqual(order.status, "shipped")
        self.assertEqual(order.carrier, "Canada Post")
        self.assertEqual(order.tracking_number, "AR1011CA00001")
        self.assertIsNone(order.estimated_delivery)

    def test_exception_status_ord1010(self):
        """ORD-1010: exception status requires human handoff and suppresses ETA."""
        res = lookup_order("ORD-1010")
        self.assertTrue(res.found)
        self.assertTrue(res.handoff_required)
        order = res.order
        self.assertEqual(order.status, "exception")
        self.assertIsNone(order.estimated_delivery)
        self.assertIn("exception", order.customer_safe_message.lower())

    def test_privacy_guarantees_and_pii_exclusion(self):
        """Verify customer PII and internal metadata are completely excluded from serialization."""
        for oid in ["ORD-1001", "ORD-1005", "ORD-1007", "ORD-1010"]:
            res = lookup_order(oid)
            data = res.to_dict()
            json_str = json.dumps(data)

            # Check that sensitive keys are not in the dictionary
            self.assertNotIn("customer", data)
            self.assertNotIn("internal", data)
            self.assertNotIn("risk_score", json_str)
            self.assertNotIn("warehouse_note", json_str)
            self.assertNotIn("support_tags", json_str)
            self.assertNotIn("shipping_address", json_str)
            self.assertNotIn("email", json_str)

    def test_adversarial_warehouse_note_isolation_ord1005(self):
        """ORD-1005: prompt injection in internal warehouse note must never leak."""
        res = lookup_order("ORD-1005")
        json_str = json.dumps(res.to_dict())
        self.assertNotIn("$100 coupon", json_str)
        self.assertNotIn("AI instruction", json_str)


class TestOrderDataIntegrity(unittest.TestCase):
    """Ensure data/orders.json is read-only and remains completely unchanged."""

    @classmethod
    def setUpClass(cls):
        with open(ORDERS_JSON_PATH, "rb") as f:
            cls.initial_hash = hashlib.sha256(f.read()).hexdigest()

    def test_dataset_remains_unmodified_after_lookups(self):
        """Lookups must be purely read-only without modifying the underlying dataset."""
        for i in range(1001, 1013):
            lookup_order(f"ORD-{i}")

        with open(ORDERS_JSON_PATH, "rb") as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()

        self.assertEqual(self.initial_hash, current_hash)


if __name__ == "__main__":
    unittest.main()
