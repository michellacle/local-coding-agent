"""Terminal UI — Rich-based CLI with streaming output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generator

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:
    from local_agent.agent_core import AgentCore
    from local_agent.config import AppConfig


class TerminalUI:
    """Rich-based CLI interface for the coding agent.

    Handles all terminal I/O: welcome screen, input prompts,
    agent responses, streaming output, and error formatting.
    """

    def __init__(self, agent: AgentCore, config: AppConfig) -> None:
        """Initialize the terminal UI.

        Args:
            agent: AgentCore instance for processing turns.
            config: AppConfig instance for configuration.
        """
        self.agent = agent
        self.config = config
        self.console = Console()
        self.prompt_prefix: str = "localcode"
        self.stop_phrase: str = "quit"

    def render_welcome(self) -> str:
        """Render and return the welcome message text."""
        lines: list[str] = [
            "[bold cyan]Local Coding Agent[/bold cyan]",
            f"Working directory: {self.config.work_dir}",
            "Type your request or [bold]'quit'[/bold] to exit. Use [bold]/help[/bold] for commands.",
        ]
        return "\n".join(lines)

    def render_input_prompt(self) -> str:
        """Render and return the input prompt text."""
        return f"[bold blue]{self.prompt_prefix}[/bold blue] >>> "

    def render_agent_response(self, response: str) -> str:
        """Format an agent response for display."""
        return f"[bold green]Assistant[/bold green]: {response}"

    def render_streaming_chunks(
        self, chunks: Generator[str, None, None]
    ) -> Generator[str, None, None]:
        """Process streaming chunks and yield formatted output.

        Args:
            chunks: Generator of text chunks from the LLM.

        Yields:
            Formatted chunk strings ready for display.
        """
        buffer: str = ""
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
        cleaned: str = raw_input.strip()
        return cleaned if cleaned else None

    def render_help(self) -> str:
        """Render help text showing available commands and usage."""
        help_lines: list[str] = [
            "[bold cyan]Local Coding Agent — Help[/bold cyan]",
            "",
            "[bold]Slash Commands:[/bold]",
            "  [yellow]/help[/yellow]       Show this help message",
            "  [yellow]/ls[/yellow] [path]    List files in working directory (or given path)",
            "  [yellow]/status[/yellow]     Show current session status",
            "  [yellow]/quit[/yellow], [yellow]/exit[/yellow]  Exit the agent",
            "",
            "[bold]What I can do:[/bold]",
            "  • [green]Read, write, and edit files[/green] in your project",
            "  • [green]Search codebases[/green] (grep filenames and content)",
            "  • [green]List directories[/green] with /ls [path]",
            "  • [green]Run shell commands[/green] and scripts",
            "  • [green]Manage git repos[/green] (init, add, commit, diff, status)",
            "  • [green]Extract PDFs[/green] to markdown text",
            "  • [green]Manage skills[/green] — create, view, and load procedural memory",
            "  • [green]Search past sessions[/green] with full-text search over history",
            "  • [green]Delegate tasks[/green] to subagents with isolated contexts",
            "  • [green]Persistent memory[/green] — save facts that survive across sessions",
            "  • [green]Ask for clarification[/green] when needed before proceeding",
            "",
            "[bold]Tips:[/bold]",
            "  • Just describe what you want — I'll figure out the steps",
            "  • I can chain multiple tools together automatically",
            "  • I adapt my retry strategy based on error type (transient vs permanent)",
            "  • If I need clarification, I'll ask before proceeding",
        ]
        return "\n".join(help_lines)

    def is_special_command(self, user_input: str) -> bool:
        """Check if the input is a special slash command."""
        return user_input.startswith("/")

    def handle_special_command(self, user_input: str) -> str | None:
        """Handle a special slash command.

        Returns:
            Response string for the command, or None if it's not a known command.
        """
        cmd: str = user_input.strip().lower()

        if cmd in ("/quit", "/exit"):
            return None  # Signal to stop

        if cmd == "/help":
            return self.render_help()

        if cmd == "/status":
            lines: list[str] = []
            lines.append(f"[bold cyan]Session Status[/bold cyan]")
            lines.append(f"  Working directory: {self.config.work_dir}")
            lines.append(f"  Model: {self.config.llm.model}")
            lines.append(f"  Toolsets: {', '.join(self.config.toolsets)}")
            return "\n".join(lines)

        if cmd == "/ls" or cmd.startswith("/ls "):
            return self._render_ls(cmd)

        return f"[bold red]Unknown command:[/bold red] {user_input}"

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

    def format_tool_call(self, tool_name: str, args: dict[str, object]) -> str:
        """Format a tool call for display."""
        args_str: str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        return f"[dim]Using tool: {tool_name}({args_str})[/dim]"

    def format_error(self, error_message: str) -> str:
        """Format an error message for display."""
        return f"[bold red]Error: {error_message}[/bold red]"

    def _render_ls(self, cmd: str) -> str:
        """Render a listing of files in the working directory.

        Supports /ls and /ls <path>.
        """
        import os
        from pathlib import Path

        # Extract optional path argument
        parts = cmd.split(maxsplit=1)
        target = parts[1] if len(parts) > 1 else self.config.work_dir

        p = Path(target).expanduser()
        if not p.exists():
            return f"[bold red]Error: path not found[/bold red] {target}"
        if not p.is_dir():
            return f"[bold red]Error: not a directory[/bold red] {target}"

        lines: list[str] = [f"[bold cyan]📁 {p.resolve()}[/bold cyan]"]

        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        for entry in entries:
            if entry.name.startswith("."):
                continue
            name = entry.name
            suffix = " [bold blue]/[/bold blue]" if entry.is_dir() else ""
            if entry.is_file():
                size = entry.stat().st_size
                if size > 1024 * 1024:
                    size_str = f"{size / (1024*1024):.1f}M"
                elif size > 1024:
                    size_str = f"{size / 1024:.1f}K"
                else:
                    size_str = f"{size}B"
                suffix = f" [dim]{size_str}[/dim]"
            lines.append(f"  {name}{suffix}")

        total = len(entries)
        hidden = sum(1 for e in entries if e.name.startswith("."))
        if total == 0:
            lines.append("  [dim](empty)[/dim]")
        else:
            lines.append(f"\n[dim]{total} entries[/dim]" + (f" [dim]({hidden} hidden)[/dim]" if hidden else ""))

        return "\n".join(lines)
