"""Terminal UI — Rich-based CLI with streaming output."""

from typing import Generator

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class TerminalUI:
    """Rich-based CLI interface for the coding agent.

    Handles all terminal I/O: welcome screen, input prompts,
    agent responses, streaming output, and error formatting.
    """

    def __init__(self, agent, config):
        """Initialize the terminal UI.

        Args:
            agent: AgentCore instance for processing turns.
            config: AppConfig instance for configuration.
        """
        self.agent = agent
        self.config = config
        self.console = Console()
        self.prompt_prefix = "You"
        self.stop_phrase = "quit"

    def render_welcome(self) -> str:
        """Render and return the welcome message text."""
        lines = [
            "[bold cyan]Local Coding Agent[/bold cyan]",
            f"Working directory: {self.config.work_dir}",
            "Type your request or [bold]'quit'[/bold] to exit.",
        ]
        return "\n".join(lines)

    def render_input_prompt(self) -> str:
        """Render and return the input prompt text."""
        return f"[bold blue]{self.prompt_prefix}[/bold blue] >>> "

    def render_agent_response(self, response: str) -> str:
        """Format an agent response for display."""
        return f"[bold green]Assistant[/bold green]: {response}"

    def render_streaming_chunks(self, chunks: Generator[str, None, None]) -> Generator[str, None, None]:
        """Process streaming chunks and yield formatted output.

        Args:
            chunks: Generator of text chunks from the LLM.

        Yields:
            Formatted chunk strings ready for display.
        """
        buffer = ""
        for chunk in chunks:
            buffer += chunk
            yield buffer
        # Final newline
        yield buffer + "\n"

    def process_input(self, raw_input: str) -> str | None:
        """Process raw user input.

        Args:
            raw_input: Raw string from the terminal.

        Returns:
            Trimmed input string, or None if empty.
        """
        cleaned = raw_input.strip()
        return cleaned if cleaned else None

    def should_stop(self, user_input: str) -> bool:
        """Check if the user wants to stop the agent.

        Args:
            user_input: The user's input string.

        Returns:
            True if the input is 'quit' or 'exit'.
        """
        return user_input.lower() in ("quit", "exit", "q")

    def run_turn(self, user_input: str) -> str:
        """Process a single agent turn.

        Args:
            user_input: The user's message.

        Returns:
            The agent's response.
        """
        return self.agent.run_turn(user_input)

    def format_tool_call(self, tool_name: str, args: dict) -> str:
        """Format a tool call for display."""
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        return f"[dim]Using tool: {tool_name}({args_str})[/dim]"

    def format_error(self, error_message: str) -> str:
        """Format an error message for display."""
        return f"[bold red]Error: {error_message}[/bold red]"
