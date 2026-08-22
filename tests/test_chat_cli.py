"""Tests for interactive chat CLI (scripts/chat.py)."""

from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.agent.orchestrator import AgentOrchestrator, AgentTurnResponse
from scripts.chat import ChatCLI, format_turn_output


class TestChatCLI(unittest.TestCase):
    """Test suite for ChatCLI formatting, commands, and session handling."""

    def setUp(self):
        self.mock_orchestrator = MagicMock(spec=AgentOrchestrator)
        self.cli = ChatCLI(orchestrator=self.mock_orchestrator)

    def test_knowledge_query_with_citations(self):
        """Knowledge response displays clean text and formatted sources list."""
        self.mock_orchestrator.process_turn.return_value = AgentTurnResponse(
            session_id="test_session",
            intent="knowledge",
            text="The standard return window is 30 days from delivery.\n\nSources:\n- 01-returns-policy-current.md — Standard return window",
            citations=["01-returns-policy-current.md — Standard return window"],
            handoff_required=False,
        )

        output, should_exit = self.cli.process_user_input("What is the return window?")

        self.assertFalse(should_exit)
        self.assertIn("Agent: The standard return window is 30 days from delivery.", output)
        self.assertIn("Sources:", output)
        self.assertIn("- 01-returns-policy-current.md — Standard return window", output)
        self.assertNotIn("Human handoff recommended.", output)

    def test_order_lookup_formatting(self):
        """Order lookup response displays status and items without leaking PII."""
        self.mock_orchestrator.process_turn.return_value = AgentTurnResponse(
            session_id="test_session",
            intent="order",
            text="Your order ORD-1001 is currently PENDING (Items: Breeze Tumbler).",
            citations=[],
            handoff_required=False,
            order_id_used="ORD-1001",
        )

        output, should_exit = self.cli.process_user_input("Where is ORD-1001?")

        self.assertFalse(should_exit)
        self.assertIn("Agent: Your order ORD-1001 is currently PENDING (Items: Breeze Tumbler).", output)
        self.assertNotIn("Sources:", output)
        self.assertNotIn("Human handoff recommended.", output)

    def test_multiturn_order_followup(self):
        """Multi-turn flow uses the same session ID across turns."""
        first_session_id = self.cli.session_id

        self.mock_orchestrator.process_turn.return_value = AgentTurnResponse(
            session_id=first_session_id,
            intent="order",
            text="Your order ORD-1007 is currently SHIPPED.",
            citations=[],
            handoff_required=False,
            order_id_used="ORD-1007",
        )

        # Turn 1
        self.cli.process_user_input("Where is ORD-1007?")
        self.mock_orchestrator.process_turn.assert_called_with(
            message="Where is ORD-1007?",
            session_id=first_session_id,
        )

        # Turn 2: Relative follow-up
        self.mock_orchestrator.process_turn.return_value = AgentTurnResponse(
            session_id=first_session_id,
            intent="order",
            text="Your order ORD-1007 is estimated to arrive on 2026-08-22.",
            citations=[],
            handoff_required=False,
            order_id_used="ORD-1007",
        )
        self.cli.process_user_input("When will it arrive?")
        self.mock_orchestrator.process_turn.assert_called_with(
            message="When will it arrive?",
            session_id=first_session_id,
        )

    def test_clear_command_resets_session(self):
        """Command /clear assigns a fresh session ID and prints reset message."""
        initial_session_id = self.cli.session_id

        output, should_exit = self.cli.process_user_input("/clear")

        self.assertFalse(should_exit)
        self.assertIn("Session cleared. Started a fresh conversation.", output)
        self.assertNotEqual(self.cli.session_id, initial_session_id)
        # Verify process_turn was not invoked on the orchestrator
        self.mock_orchestrator.process_turn.assert_not_called()

    def test_help_command(self):
        """Command /help returns available commands."""
        output, should_exit = self.cli.process_user_input("/help")

        self.assertFalse(should_exit)
        self.assertIn("/help", output)
        self.assertIn("/clear", output)
        self.assertIn("/exit", output)
        self.mock_orchestrator.process_turn.assert_not_called()

    def test_exit_command(self):
        """Command /exit signals termination."""
        output, should_exit = self.cli.process_user_input("/exit")

        self.assertTrue(should_exit)
        self.assertEqual(output, "Goodbye!")

    def test_handoff_display(self):
        """Conflict or unknown case displays human handoff recommendation."""
        self.mock_orchestrator.process_turn.return_value = AgentTurnResponse(
            session_id="test_session",
            intent="knowledge",
            text="Documentation contains conflicting cleaning guidance.",
            citations=[
                "12-breeze-tumbler-product-card.md — Cleaning",
                "11-product-care.md — Breeze Tumbler",
            ],
            handoff_required=True,
        )

        output, should_exit = self.cli.process_user_input("Can I put the Breeze Tumbler in the dishwasher?")

        self.assertFalse(should_exit)
        self.assertIn("Human handoff recommended.", output)
        self.assertIn("Sources:", output)
        self.assertIn("- 12-breeze-tumbler-product-card.md — Cleaning", output)

    def test_empty_input_ignored(self):
        """Whitespace or empty input produces no output and does not exit."""
        output, should_exit = self.cli.process_user_input("   ")
        self.assertIsNone(output)
        self.assertFalse(should_exit)


if __name__ == "__main__":
    unittest.main()
