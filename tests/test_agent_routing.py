"""Unit tests for deterministic routing and bounded session context."""

import unittest

from app.agent.router import RouteDecision, Router, route_query
from app.agent.session import SessionContext, Turn


class TestSessionContext(unittest.TestCase):
    """Test session state isolation, updating, and bounded memory limits."""

    def test_bounded_recent_turns_limit(self):
        """Verify adding more than 5 turns keeps strictly the 5 most recent turns."""
        session = SessionContext(session_id="sess_123", max_turns=5)

        for i in range(1, 11):
            session.add_turn("user" if i % 2 != 0 else "assistant", f"Turn {i}")

        self.assertEqual(len(session.recent_turns), 5)
        self.assertEqual(session.recent_turns[0].text, "Turn 6")
        self.assertEqual(session.recent_turns[-1].text, "Turn 10")

    def test_session_isolation_no_cross_contamination(self):
        """Verify two session objects remain completely isolated."""
        sess_a = SessionContext(session_id="sess_a")
        sess_b = SessionContext(session_id="sess_b")

        sess_a.set_order_context("ORD-1007")
        sess_a.add_turn("user", "Hello from A")

        sess_b.set_order_context("ORD-1001")
        sess_b.add_turn("user", "Hello from B")

        self.assertEqual(sess_a.last_order_id, "ORD-1007")
        self.assertEqual(sess_b.last_order_id, "ORD-1001")
        self.assertEqual(sess_a.recent_turns[0].text, "Hello from A")
        self.assertEqual(sess_b.recent_turns[0].text, "Hello from B")


class TestDeterministicRouting(unittest.TestCase):
    """Test deterministic rule-based query routing."""

    def test_explicit_uppercase_order_id(self):
        dec = route_query("Where is ORD-1007?")
        self.assertEqual(dec.intent, "order")
        self.assertEqual(dec.normalized_order_id, "ORD-1007")
        self.assertFalse(dec.is_follow_up)

    def test_lowercase_order_id(self):
        dec = route_query("status for ord-1007")
        self.assertEqual(dec.intent, "order")
        self.assertEqual(dec.normalized_order_id, "ORD-1007")

    def test_separator_variation_order_id(self):
        dec = route_query("Check order ORD 1007 please")
        self.assertEqual(dec.intent, "order")
        self.assertEqual(dec.normalized_order_id, "ORD-1007")

    def test_malformed_order_like_string(self):
        dec = route_query("Please check ORD-XYZ")
        self.assertEqual(dec.intent, "order")
        self.assertIsNone(dec.normalized_order_id)
        self.assertEqual(dec.reason, "malformed_order_id")

    def test_order_intent_missing_id(self):
        """'Where is my order?' without active session order ID routes to order with None ID."""
        session = SessionContext(session_id="sess_empty")
        dec = route_query("Where is my order?", session=session)
        self.assertEqual(dec.intent, "order")
        self.assertIsNone(dec.order_id)
        self.assertIsNone(dec.normalized_order_id)
        self.assertFalse(dec.is_follow_up)
        self.assertEqual(dec.reason, "order_intent_missing_id")

    def test_order_follow_up_with_session_context(self):
        """'When will it arrive?' with active session order resolves to session order ID."""
        session = SessionContext(session_id="sess_ord")
        session.set_order_context("ORD-1007")

        dec = route_query("When will it arrive?", session=session)
        self.assertEqual(dec.intent, "order")
        self.assertEqual(dec.normalized_order_id, "ORD-1007")
        self.assertTrue(dec.is_follow_up)
        self.assertEqual(dec.reason, "session_order_follow_up")

    def test_topic_switch_order_to_warranty_policy(self):
        """General warranty query following an order lookup routes to knowledge without order mixing."""
        session = SessionContext(session_id="sess_mix")
        session.set_order_context("ORD-1007")

        dec = route_query("What is your warranty policy?", session=session)
        self.assertEqual(dec.intent, "knowledge")
        self.assertEqual(dec.topic, "warranty")
        self.assertFalse(dec.is_follow_up)

    def test_order_follow_up_after_intermediate_warranty_query(self):
        """'When will it arrive?' after intermediate warranty query recovers stored ORD-1007."""
        session = SessionContext(session_id="sess_turn3")
        session.set_order_context("ORD-1007")
        # Intermediate topic was warranty
        session.set_topic("warranty")

        dec = route_query("When will it arrive?", session=session)
        self.assertEqual(dec.intent, "order")
        self.assertEqual(dec.normalized_order_id, "ORD-1007")
        self.assertTrue(dec.is_follow_up)

    def test_international_shipping_query(self):
        dec = route_query("Do you ship internationally?")
        self.assertEqual(dec.intent, "knowledge")
        self.assertEqual(dec.topic, "international_shipping")

    def test_international_shipping_follow_up_canada(self):
        session = SessionContext(session_id="sess_intl")
        session.set_topic("international_shipping")

        dec = route_query("What about Canada?", session=session)
        self.assertEqual(dec.intent, "knowledge")
        self.assertEqual(dec.topic, "international_shipping")
        self.assertEqual(dec.entity, "Canada")
        self.assertTrue(dec.is_follow_up)

    def test_return_policy_trailplus_follow_up(self):
        session = SessionContext(session_id="sess_ret")
        session.set_topic("return_policy")

        dec = route_query("What about TrailPlus members?", session=session)
        self.assertEqual(dec.intent, "knowledge")
        self.assertEqual(dec.topic, "return_policy")
        self.assertEqual(dec.entity, "TrailPlus")
        self.assertTrue(dec.is_follow_up)

    def test_greetings_routing(self):
        for greeting in ["Hello", "Hi!", "Hey there", "Good morning", "howdy"]:
            dec = route_query(greeting)
            self.assertEqual(dec.intent, "greeting", f"Failed on '{greeting}'")

    def test_ambiguous_unknown_query(self):
        for unknown_text in ["asdfghjkl123", "quantum banana flux"]:
            dec = route_query(unknown_text)
            self.assertEqual(dec.intent, "unknown", f"Failed on '{unknown_text}'")


if __name__ == "__main__":
    unittest.main()
