"""Interactive CLI for the Aster & Row Customer Support Agent."""

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

    def __init__(self, orchestrator: Optional[AgentOrchestrator] = None) -> None:
        if orchestrator is None:
            vector_store = VectorStore.load()
            order_service = OrderService()
            orchestrator = AgentOrchestrator(
                vector_store=vector_store,
                order_service=order_service,
            )
        self.orchestrator = orchestrator
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

        response = self.orchestrator.process_turn(
            message=text,
            session_id=self.session_id,
        )
        return format_turn_output(response), False

    def run(self) -> None:
        """Start the interactive REPL loop."""
        print("Aster & Row Support Agent")
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
    cli = ChatCLI()
    cli.run()


if __name__ == "__main__":
    main()
