"""Interactive CLI for the Aster & Row Customer Support Agent."""

import argparse
from pathlib import Path
import sys
import uuid
from typing import Optional, Tuple

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.agent.orchestrator import AgentOrchestrator, AgentTurnResponse
from app.orders.service import OrderService
from app.rag.vector_store import VectorStore
from evaluation.run_evaluation import MockGeminiEvaluatorClient


def format_turn_output(response: AgentTurnResponse) -> str:
    """Format AgentTurnResponse for clean CLI display."""
    lines = []
    # Strip any duplicated Sources section embedded in the LLM text body
    clean_text = response.text.split("\n\nSources:")[0].strip()
    lines.append(f"Agent: {clean_text}")

    if response.citations:
        lines.append("")
        lines.append("Sources:")
        for citation in response.citations:
            lines.append(f"- {citation}")

    if response.handoff_required:
        lines.append("")
        lines.append("Human handoff recommended.")

    return "\n".join(lines)


class ChatCLI:
    """CLI session manager."""

    def __init__(self, orchestrator: Optional[AgentOrchestrator] = None, use_mock: bool = False) -> None:
        if orchestrator is None:
            vector_store = VectorStore.load()
            order_service = OrderService()
            gemini_client = MockGeminiEvaluatorClient() if use_mock else None
            orchestrator = AgentOrchestrator(
                vector_store=vector_store,
                order_service=order_service,
                gemini_client=gemini_client,
            )
        self.orchestrator = orchestrator
        self.use_mock = use_mock
        self.session_id = self._generate_session_id()

    @staticmethod
    def _generate_session_id() -> str:
        return f"cli_{uuid.uuid4().hex[:8]}"

    def reset_session(self) -> None:
        """Reset the conversation state by assigning a new session ID."""
        self.session_id = self._generate_session_id()

    def process_user_input(self, user_input: str) -> Tuple[Optional[str], bool]:
        """Process a single line of user input.

        Returns:
            (output_text, should_exit)
        """
        text = user_input.strip()
        if not text:
            return None, False

        if text.lower() in ("/exit", "/quit"):
            return "Goodbye!", True

        if text.lower() == "/help":
            help_text = (
                "Available commands:\n"
                "  /help   - Show this help message\n"
                "  /clear  - Clear conversation history and start a fresh session\n"
                "  /exit   - Exit the chat session"
            )
            return help_text, False

        if text.lower() == "/clear":
            self.reset_session()
            return "Session cleared. Started a fresh conversation.", False

        try:
            response = self.orchestrator.process_turn(
                message=text,
                session_id=self.session_id,
            )
            return format_turn_output(response), False
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "resource_exhausted" in err_msg or "quota" in err_msg:
                return (
                    "Agent: Live Gemini free-tier quota is currently exhausted (HTTP 429). "
                    "You can run with 'python scripts/chat.py --mock' for offline evaluation.\n\n"
                    "Human handoff recommended.",
                    False,
                )
            return f"Agent: An error occurred while processing your request: {e}\n\nHuman handoff recommended.", False

    def run(self) -> None:
        """Start the interactive REPL loop."""
        mode_str = " (Mock Mode)" if self.use_mock else ""
        print(f"Aster & Row Support Agent{mode_str}")
        print("Type /help for commands.\n")

        while True:
            try:
                user_input = input("You: ")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            output, should_exit = self.process_user_input(user_input)
            if output is not None:
                print(output)
                print()

            if should_exit:
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Aster & Row Support Agent Interactive CLI")
    parser.add_argument("--mock", "-m", action="store_true", help="Run in deterministic mock mode without live Gemini API quota")
    args = parser.parse_args()

    cli = ChatCLI(use_mock=args.mock)
    cli.run()


if __name__ == "__main__":
    main()
