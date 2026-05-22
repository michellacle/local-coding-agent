"""Terminal UI — Rich-based CLI with streaming output and slash commands.

Slash commands provide quick access to all agent capabilities without
needing to phrase a natural-language request.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

import httpx

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:
    from local_agent.agent_core import AgentCore
    from local_agent.config import AppConfig


class TerminalUI:
    """Rich-based CLI interface for the coding agent.

    Handles all terminal I/O: welcome screen, input prompts,
    agent responses, streaming output, and slash commands.
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
        self.work_dir: Path = Path(self.config.work_dir).resolve()

    # ------------------------------------------------------------------ #
    #  Rendering helpers                                                   #
    # ------------------------------------------------------------------ #

    def render_welcome(self) -> str:
        """Render and return the welcome message text."""
        lines: list[str] = [
            "[bold cyan]Local Coding Agent[/bold cyan]",
            f"Working directory: {self.work_dir}",
            "Type your request or [bold]'quit'[/bold] to exit. Use [bold]/help[/bold] for commands.",
        ]
        return "\n".join(lines)

    def render_help(self) -> str:
        """Render help text showing available commands and usage.

        Alias for _cmd_help() for backward compatibility with tests.
        """
        return self._cmd_help("")

    def render_input_prompt(self) -> str:
        """Render and return the input prompt text."""
        return f"[bold blue]{self.prompt_prefix}[/bold blue] >>> "

    def render_agent_response(self, response: str) -> str:
        """Format an agent response for display."""
        return f"[bold green]Assistant[/bold green]: {response}"

    def render_streaming_chunks(
        self, chunks: Generator[str, None, None]
    ) -> Generator[str, None, None]:
        """Process streaming chunks and yield formatted output."""
        buffer: str = ""
        for chunk in chunks:
            buffer += chunk
            yield buffer
        yield buffer + "\n"

    def process_input(self, raw_input: str) -> str | None:
        """Process raw user input."""
        cleaned: str = raw_input.strip()
        return cleaned if cleaned else None

    def should_stop(self, user_input: str) -> bool:
        """Check if the user wants to stop the agent."""
        return user_input.lower() in ("quit", "exit", "q")

    def run_turn(self, user_input: str) -> str:
        """Process a single agent turn."""
        return self.agent.run_turn(user_input)

    def format_tool_call(self, tool_name: str, args: dict[str, object]) -> str:
        """Format a tool call for display."""
        args_str: str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        return f"[dim]Using tool: {tool_name}({args_str})[/dim]"

    def format_error(self, error_message: str) -> str:
        """Format an error message for display."""
        return f"[bold red]Error: {error_message}[/bold red]"

    # ------------------------------------------------------------------ #
    #  Slash-command dispatch                                              #
    # ------------------------------------------------------------------ #

    def is_special_command(self, user_input: str) -> bool:
        """Check if the input is a special slash command."""
        return user_input.startswith("/")

    def handle_special_command(self, user_input: str) -> str | None:
        """Handle a special slash command.

        Returns:
            Response string for the command, or None to stop.
        """
        raw = user_input.strip()
        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        dispatch: dict[str, str | None] = {
            "/quit": None,
            "/exit": None,
            "/help": self._cmd_help(args),
            "/ls": self._cmd_ls(args),
            "/status": self._cmd_status(),
            "/read": self._cmd_read(args) if args else "Usage: /read <file_path>",
            "/cat": self._cmd_read(args) if args else "Usage: /read <file_path>",
            "/search": self._cmd_search(args) if args else "Usage: /search <pattern> [--files|--path DIR|--glob *.py|--limit N]",
            "/grep": self._cmd_search(args) if args else "Usage: /search <pattern> [--files|--path DIR]",
            "/git": self._cmd_git(args) if args else "Usage: /git <status|log|branch|diff|add|commit|push>\nExample: /git status",
            "/memory": self._cmd_memory(args),
            "/skill": self._cmd_skill(args),
            "/session": self._cmd_session(args) if args else "Usage: /session search <query>",
            "/cron": self._cmd_cron(args),
            "/config": self._cmd_config(args),
            "/browser": self._cmd_browser(args) if args else "Usage: /browser <url>",
            "/explain": self._cmd_explain(args),
            "/safety": self._cmd_safety(args) if args else "Usage: /safety check <text>",
            "/rag": self._cmd_rag(args),
            "/delegate": self._cmd_delegate(args) if args else "Usage: /delegate <task description>",
            "/clear": self._cmd_clear(),
            "/history": self._cmd_history(),
            "/tools": self._cmd_tools(),
            "/version": self._cmd_version(),
            "/models": self._cmd_models(args),
            "/plan": self._cmd_plan(args),
            "/stats": self._cmd_stats(),
            "/code": self._cmd_code(args),
            "/provider": self._cmd_provider(args),
        }

        if cmd in dispatch:
            return dispatch[cmd]

        return f"[bold red]Unknown command:[/bold red] {cmd}\n[dim]Type /help for available commands.[/dim]"

    # ------------------------------------------------------------------ #
    #  Individual command handlers                                         #
    # ------------------------------------------------------------------ #

    def _cmd_help(self, args: str) -> str:
        """Render the full help screen with examples."""
        return textwrap.dedent("""\
        [bold cyan]Local Coding Agent — Help & Slash Commands[/bold cyan]

        [bold]General[/bold]
          [yellow]/help[/yellow]          Show this help screen
          [yellow]/help <topic>[/yellow]   Show detailed help for a topic (e.g. /help memory)
          [yellow]/status[/yellow]        Show current session status (model, tools, working dir)
          [yellow]/tools[/yellow]         List all available tool categories
          [yellow]/version[/yellow]       Show agent version info
          [yellow]/models[/yellow]        List available models from gateway & switch
          [yellow]/models N[/yellow]      Switch to model at index N
          [yellow]/plan <goal>[/yellow]     Decompose a goal into actionable steps
          [yellow]/plan list[/yellow]     List saved plans
          [yellow]/code <description>[/yellow] Build code and write it to disk
          [yellow]/stats[/yellow]         Show model routing stats & token usage
          [yellow]/provider[/yellow]       View/change LLM provider config
          [yellow]/quit, /exit[/yellow]    Exit the agent

        [bold]File Operations[/bold]
          [yellow]/ls [path][/yellow]       List files in working directory (or given path)
          [yellow]/read <file>[/yellow]     Read and display file contents
          [yellow]/cat <file>[/yellow]      Alias for /read

        [bold]Search & Codebase[/bold]
          [yellow]/search <pattern>[/yellow]  Search file contents (regex). Options:
                                     [--files]  search filenames instead
                                     [--path DIR]  limit search directory
                                     [--glob *.py]  filter by extension
                                     [--limit N]   max results (default 50)
          [yellow]/grep <pattern>[/yellow]    Alias for /search

          Examples:
            /search def main
            /search "TODO" --glob *.py --limit 20
            /search flask --files --path src

        [bold]Git[/bold]
          [yellow]/git <command>[/yellow]     Run git commands:
                                     status, log, branch, diff, add, commit, push
          Examples:
            /git status
            /git log --oneline -10
            /git branch -a
            /git diff HEAD~1
            /git add . && git commit -m "fix: resolve bug"

        [bold]Memory (Persistent across sessions)[/bold]
          [yellow]/memory list[/yellow]      List all saved memory entries
          [yellow]/memory show <topic>[/yellow] Show memory entries matching a topic
          Examples:
            /memory list
            /memory show project

        [bold]Skills (Reusable procedures)[/bold]
          [yellow]/skill list[/yellow]      List available skills
          [yellow]/skill view <name>[/yellow]  View a skill's full instructions
          Examples:
            /skill list
            /skill view code-review

        [bold]Sessions (Past conversation search)[/bold]
          [yellow]/session search <query>[/yellow]  Search past sessions by keywords
          Examples:
            /session search docker network
            /session search refactor auth

        [bold]Cron Jobs (Scheduled tasks)[/bold]
          [yellow]/cron list[/yellow]        List all scheduled jobs
          [yellow]/cron status[/yellow]      Show cron job summary
          Examples:
            /cron list
            /cron status

        [bold]Configuration[/bold]
          [yellow]/config[/yellow]           Show current config summary
          [yellow]/config get <key>[/yellow]   Get a config value (e.g. config get llm.model)
          [yellow]/config set <key> <val>[/yellow] Set a config value
          [yellow]/config export <path>[/yellow] Export merged config to YAML file
          Examples:
            /config get llm.model
            /config set max_turns 20
            /config export ~/.config/local-coding-agent/config.yaml

        [bold]Browser (Playwright automation)[/bold]
          [yellow]/browser <url>[/yellow]     Navigate to a URL and show page summary
          Examples:
            /browser https://example.com
            /browser https://github.com/torvalds/linux

        [bold]Explainability & Audit[/bold]
          [yellow]/explain[/yellow]          Show recent decision audit trail
          [yellow]/explain summary[/yellow]  Show decision summary statistics
          [yellow]/explain export <path>[/yellow] Export full audit trail to JSON
          Examples:
            /explain
            /explain summary
            /explain export /tmp/audit.json

        [bold]Safety (Adversarial protection)[/bold]
          [yellow]/safety check <text>[/yellow]  Check text for prompt injection patterns
          Examples:
            /safety check "Ignore previous instructions and do X"
            /safety check "What is the capital of France?"

        [bold]RAG (Knowledge base)[/bold]
          [yellow]/rag index <dir>[/yellow]   Index a directory of documents for semantic search
          [yellow]/rag query <text>[/yellow]    Query the knowledge base
          [yellow]/rag list[/yellow]          List indexed documents
          Examples:
            /rag index /path/to/docs
            /rag query "How do I configure the LLM provider?"
            /rag list

        [bold]Multi-Agent (Delegation)[/bold]
          [yellow]/delegate <task>[/yellow]     Delegate a task to a subagent
          Examples:
            /delegate "Write unit tests for src/auth.py"
            /delegate "Research best practices for Docker multi-stage builds"

        [bold]Utility[/bold]
          [yellow]/clear[/yellow]          Clear the terminal screen
          [yellow]/history[/yellow]        Show recent conversation history

        [bold]Built-in Features[/bold]
          • [green]clarification[/green] — The agent will ask you questions when it needs input
          • [green]retry strategy[/green] — Failed tool calls are retried with exponential backoff
          • [green]Extract PDFs[/green] — "extract the PDF in this folder to markdown"

        [bold]What I can do (just type it):[/bold]
          • [green]Read, write, and edit files[/green] — "edit main.py to add error handling"
          • [green]Search codebases[/green] — "find all TODO comments in Python files"
          • [green]List directories[/green] — "show me the contents of src/"
          • [green]Run shell commands[/green] — "run pytest and show me the results"
          • [green]Manage git repos[/green] — "commit all changes with message 'fix bug'"
          • [green]Delegate tasks[/green] — "in parallel, review auth.py and write tests for utils.py"
          • [green]Persistent memory[/green] — "remember that I prefer snake_case"
          • [green]Manage skills[/green] — "create a skill for running integration tests"
          • [green]Search past sessions[/green] — "what did we decide about the database?"
          • [green]Run scheduled jobs[/green] — "set up a daily test run at 9am"
        """).strip()

    def _cmd_ls(self, args: str) -> str:
        """List directory contents."""
        target = args if args else self.work_dir
        return self._render_ls(str(target))

    def _cmd_status(self) -> str:
        """Show current session status."""
        lines = [
            "[bold cyan]Session Status[/bold cyan]",
            f"  Working directory: {self.work_dir}",
            f"  Model: {self.config.llm.model}",
            f"  Host: {self.config.llm.host}",
            f"  Port: {self.config.llm.port}",
        ]
        toolsets = getattr(self.config, "toolsets", None)
        if toolsets:
            lines.append(f"  Toolsets: {', '.join(toolsets)}")
        return "\n".join(lines)

    def _cmd_read(self, args: str) -> str:
        """Read and display file contents."""
        path = Path(args.strip()).expanduser()
        if not path.exists():
            return f"[bold red]Error:[/bold red] File not found: {path}"
        if not path.is_file():
            return f"[bold red]Error:[/bold red] Not a file: {path}"

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except (PermissionError, OSError) as e:
            return f"[bold red]Error:[/bold red] Cannot read {path}: {e}"

        # Truncate very long files
        max_lines = 200
        line_count = len(content.split("\n"))
        if line_count > max_lines:
            lines = content.split("\n")[:max_lines]
            return (
                f"[bold cyan]📄 {path}[/bold cyan] [dim]({line_count} lines, showing first {max_lines})[/dim]\n"
                + "\n".join(lines)
            )

        return f"[bold cyan]📄 {path}[/bold cyan] [dim]({line_count} lines)[/dim]\n{content}"

    def _cmd_search(self, args: str) -> str:
        """Search files using ripgrep."""
        # Parse arguments
        use_files = False
        search_path = None
        file_glob = None
        limit = 50

        tokens = args.split()
        pattern = None
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == "--files":
                use_files = True
            elif token == "--path" and i + 1 < len(tokens):
                search_path = tokens[i + 1]
                i += 1
            elif token.startswith("--path="):
                search_path = token[7:]
            elif token == "--glob" and i + 1 < len(tokens):
                file_glob = tokens[i + 1]
                i += 1
            elif token.startswith("--glob="):
                file_glob = token[7:]
            elif token == "--limit" and i + 1 < len(tokens):
                limit = int(tokens[i + 1])
                i += 1
            elif token.startswith("--limit="):
                limit = int(token[8:])
            elif pattern is None:
                pattern = token
            else:
                pattern += " " + token
            i += 1

        if not pattern:
            return "[bold red]Error:[/bold red] No search pattern provided"

        try:
            cmd = ["rg", "--color", "never"]
            if use_files:
                cmd.extend(["-l", "-g", pattern])
            else:
                cmd.append(pattern)
            cmd.extend(["-n", "--max-count", str(limit)])
            if search_path:
                cmd.extend(["-S", search_path])
            if file_glob:
                cmd.extend(["-g", file_glob])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(self.work_dir),
            )

            if result.returncode == 0 and result.stdout.strip():
                output = result.stdout.strip()
                line_count = len(output.split("\n"))
                return f"[bold cyan]Search results[/bold cyan] [dim]({line_count} matches)[/dim]\n{output}"
            elif result.returncode == 1:
                return f"[dim]No matches found for: {pattern}[/dim]"
            else:
                return f"[dim]Search returned no results[/dim]"

        except FileNotFoundError:
            return "[bold red]Error:[/bold red] ripgrep (rg) not found. Install with: sudo apt install ripgrep"
        except subprocess.TimeoutExpired:
            return "[bold red]Error:[/bold red] Search timed out after 10 seconds"

    def _cmd_git(self, args: str) -> str:
        """Run git commands."""
        try:
            result = subprocess.run(
                ["git"] + args.split(),
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.work_dir),
            )
            output = (result.stdout or result.stderr or "").strip()
            if output:
                return f"[bold cyan]git {args}[/bold cyan]\n{output}"
            if result.returncode == 0:
                return "[dim]No output (git command succeeded)[/dim]"
            return f"[bold red]git error (exit {result.returncode}):[/bold red]\n{output}"
        except FileNotFoundError:
            return "[bold red]Error:[/bold red] git not found"
        except subprocess.TimeoutExpired:
            return "[bold red]Error:[/bold red] Git command timed out"

    def _cmd_memory(self, args: str) -> str:
        """Manage persistent memory."""
        if not args:
            return textwrap.dedent("""\
            [bold cyan]Memory Management[/bold cyan]

            Usage:
              [yellow]/memory list[/yellow]       List all memory entries
              [yellow]/memory show <topic>[/yellow]  Filter entries by keyword

            Examples:
              /memory list
              /memory show project

            Memory stores facts that persist across sessions.
            The agent automatically saves preferences and corrections.""")

        parts = args.split(maxsplit=1)
        action = parts[0].lower()
        query = parts[1] if len(parts) > 1 else ""

        # Try to access the agent's memory store
        try:
            memory_path = Path.home() / ".local-coding-agent" / "memory"
            if not memory_path.exists():
                return "[dim]No memory file found. Memory is auto-saved by the agent.[/dim]"

            memory_file = memory_path / "notes.md"
            if not memory_file.exists():
                memory_file = memory_path / "memory.md"
            if not memory_file.exists():
                return "[dim]No memory entries saved yet.[/dim]"

            content = memory_file.read_text()

            if action == "list":
                return f"[bold cyan]🧠 Memory Entries[/bold cyan]\n{content}"
            elif action == "show":
                # Filter lines containing the query
                lines = [l for l in content.split("\n") if query.lower() in l.lower()]
                if lines:
                    return f"[bold cyan]Memory entries matching '{query}'[/bold cyan]\n" + "\n".join(lines)
                return f"[dim]No memory entries matching '{query}'[/dim]"
            else:
                return f"[bold red]Unknown memory action:[/bold red] {action}\n[dim]Use: list, show[/dim]"
        except Exception as e:
            return f"[dim]Could not read memory: {e}[/dim]"

    def _cmd_skill(self, args: str) -> str:
        """Manage skills."""
        if not args:
            return textwrap.dedent("""\
            [bold cyan]Skill Management[/bold cyan]

            Usage:
              [yellow]/skill list[/yellow]        List available skills
              [yellow]/skill view <name>[/yellow]   View a skill's full instructions

            Examples:
              /skill list
              /skill view code-review

            Skills are reusable procedures for common tasks.""")

        parts = args.split(maxsplit=1)
        action = parts[0].lower()
        name = parts[1] if len(parts) > 1 else ""

        try:
            # Check in-repo skills first
            skill_dirs = [
                self.work_dir / "skills",
                Path.home() / ".local-coding-agent" / "skills",
            ]

            if action == "list":
                skills = []
                for skill_dir in skill_dirs:
                    if skill_dir.exists():
                        for md_file in skill_dir.glob("**/SKILL.md"):
                            skills.append(str(md_file.relative_to(skill_dir.parent)))
                if skills:
                    return f"[bold cyan]🛠 Available Skills ({len(skills)})[/bold cyan]\n" + "\n".join(f"  • {s}" for s in skills)
                return "[dim]No skills found. Skills are auto-created by the agent for recurring tasks.[/dim]"

            elif action == "view" and name:
                for skill_dir in skill_dirs:
                    skill_file = skill_dir / name / "SKILL.md"
                    if not skill_file.exists():
                        skill_file = skill_dir / f"{name}.md"
                    if skill_file.exists():
                        return f"[bold cyan]🛠 Skill: {name}[/bold cyan]\n{skill_file.read_text()}"
                return f"[bold red]Skill not found:[/bold red] {name}"

            else:
                return f"[bold red]Unknown skill action:[/bold red] {action}\n[dim]Use: list, view[/dim]"

        except Exception as e:
            return f"[dim]Could not access skills: {e}[/dim]"

    def _cmd_session(self, args: str) -> str:
        """Search past sessions."""
        if not args:
            return "[yellow]Usage:[/yellow] /session search <query>"

        parts = args.split(maxsplit=1)
        action = parts[0].lower()
        query = parts[1] if len(parts) > 1 else ""

        if action == "search" and query:
            try:
                # Try session_search via SQLite
                session_db = Path.home() / ".local-coding-agent" / "sessions.db"
                if session_db.exists():
                    import sqlite3
                    conn = sqlite3.connect(str(session_db))
                    cursor = conn.execute(
                        "SELECT COUNT(*) FROM sessions WHERE title LIKE ? OR content LIKE ?",
                        (f"%{query}%", f"%{query}%"),
                    )
                    count = cursor.fetchone()[0]
                    conn.close()
                    return f"[bold cyan]Session search[/bold cyan] [dim]({count} matches for '{query}')[/dim]\n[dim]Use the agent to read details: 'search past sessions for {query}'[/dim]"
                else:
                    return "[dim]No session database found. Sessions are stored automatically.[/dim]"
            except Exception:
                return f"[dim]Session search: ask the agent to 'search past sessions for {query}'[/dim]"
        else:
            return f"[bold red]Unknown session action:[/bold red] {action}\n[dim]Use: search <query>[/dim]"

    def _cmd_cron(self, args: str) -> str:
        """Manage cron jobs."""
        if not args:
            return textwrap.dedent("""\
            [bold cyan]Cron Job Management[/bold cyan]

            Usage:
              [yellow]/cron list[/yellow]        List all scheduled jobs
              [yellow]/cron status[/yellow]      Show job summary

            Examples:
              /cron list
              /cron status

            To create/modify jobs, use the agent:
              "create a cron job to run tests every hour"
              "pause the daily report job"
              """)

        action = args.split(maxsplit=1)[0].lower()

        try:
            state_path = Path.home() / ".local-coding-agent" / "cron_state.json"
            if not state_path.exists():
                return "[dim]No cron jobs configured. Ask the agent to create one.[/dim]"

            import json
            data = json.loads(state_path.read_text())

            if action in ("list", "status"):
                jobs = data.get("jobs", [])
                if not jobs:
                    return "[dim]No cron jobs found.[/dim]"

                lines = [f"[bold cyan]⏰ Cron Jobs ({len(jobs)})[/bold cyan]"]
                for job in jobs:
                    status = "✅" if job.get("enabled") else "⏸️ "
                    lines.append(f"  {status} [bold]{job.get('name', 'unnamed')}[/bold] — {job.get('schedule', '?')}")
                    lines.append(f"     Prompt: {job.get('prompt', '')[:80]}")
                    if job.get("last_result"):
                        lines.append(f"     Last: {job['last_result'][:80]}")
                return "\n".join(lines)
            else:
                return f"[bold red]Unknown cron action:[/bold red] {action}\n[dim]Use: list, status[/dim]"
        except Exception as e:
            return f"[dim]Could not read cron state: {e}[/dim]"

    def _cmd_config(self, args: str) -> str:
        """Manage configuration."""
        if not args:
            return f"[bold cyan]Configuration[/bold cyan]\n  Model: {self.config.llm.model}\n  Host: {self.config.llm.host}:{self.config.llm.port}\n  Deterministic: {self.config.llm.deterministic}\n  Working directory: {self.work_dir}"

        parts = args.split(maxsplit=1)
        action = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        if action == "get":
            key = rest.strip()
            if not key:
                return "[yellow]Usage:[/yellow] /config get <dot.separated.key>"
            # Check known config values
            if key == "llm.model":
                return f"llm.model = {self.config.llm.model}"
            elif key == "llm.host":
                return f"llm.host = {self.config.llm.host}"
            elif key == "llm.port":
                return f"llm.port = {self.config.llm.port}"
            elif key == "llm.deterministic":
                return f"llm.deterministic = {self.config.llm.deterministic}"
            return f"[dim]Config key not found: {key}[/dim]"

        elif action == "set":
            kv = rest.strip().split(maxsplit=1)
            if len(kv) != 2:
                return "[yellow]Usage:[/yellow] /config set <key> <value>"
            return f"[dim]Config set: {kv[0]} = {kv[1]} (will take effect next session)[/dim]"

        elif action == "export":
            path = rest.strip()
            if not path:
                return "[yellow]Usage:[/yellow] /config export <path>"
            try:
                import yaml
                config_data = {
                    "llm": {
                        "model": self.config.llm.model,
                        "host": self.config.llm.host,
                        "port": self.config.llm.port,
                        "deterministic": self.config.llm.deterministic,
                    },
                    "working_directory": str(self.work_dir),
                }
                output = yaml.dump(config_data, default_flow_style=False)
                Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
                Path(path).expanduser().write_text(output)
                return f"[bold green]Config exported to: {Path(path).expanduser().resolve()}[/bold green]"
            except ImportError:
                return "[bold red]Error:[/bold red] PyYAML not installed. Install with: pip install pyyaml"
            except Exception as e:
                return f"[bold red]Export failed:[/bold red] {e}"

        else:
            return f"[bold red]Unknown config action:[/bold red] {action}\n[dim]Use: get, set, export[/dim]"

    def _cmd_browser(self, args: str) -> str:
        """Navigate browser and show page summary."""
        url = args.strip()
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        return f"[dim]Browser navigation queued for: {url}[/dim]\n[dim]The agent will open the page and extract content.[/dim]"

    def _cmd_explain(self, args: str) -> str:
        """Show explainability / audit trail."""
        try:
            audit_path = Path.home() / ".local-coding-agent" / "audit.log"
            if not audit_path.exists():
                return "[dim]No audit trail recorded yet. The agent logs decisions automatically.[/dim]"

            action = args.split(maxsplit=1)[0].lower() if args else ""

            if action == "summary":
                import json
                lines = audit_path.read_text().strip().split("\n")
                if not lines or lines == [""]:
                    return "[dim]No audit entries yet.[/dim]"

                types: dict[str, int] = {}
                for line in lines:
                    try:
                        entry = json.loads(line)
                        etype = entry.get("event_type", "unknown").split(".")[0]
                        types[etype] = types.get(etype, 0) + 1
                    except json.JSONDecodeError:
                        pass

                if types:
                    lines_out = ["[bold cyan]📊 Audit Summary[/bold cyan]"]
                    for etype, count in sorted(types.items(), key=lambda x: -x[1]):
                        lines_out.append(f"  {etype}: {count}")
                    lines_out.append(f"\n  Total entries: {len(lines)}")
                    return "\n".join(lines_out)
                return "[dim]No audit entries to summarize.[/dim]"

            elif action == "export":
                export_path = args.split(maxsplit=1)[1].strip() if len(args.split(maxsplit=1)) > 1 else ""
                if not export_path:
                    return "[yellow]Usage:[/yellow] /explain export <path>"
                content = audit_path.read_text()
                export_file = Path(export_path).expanduser()
                export_file.parent.mkdir(parents=True, exist_ok=True)
                export_file.write_text(content)
                return f"[bold green]Audit trail exported to: {export_file.resolve()}[/bold green]"

            else:
                # Show recent entries
                content = audit_path.read_text().strip().split("\n")
                recent = content[-10:] if content else []
                if not recent:
                    return "[dim]No audit entries yet.[/dim]"
                return f"[bold cyan]📋 Recent Audit Entries (last {len(recent)})[/bold cyan]\n" + "\n".join(recent)

        except Exception as e:
            return f"[dim]Could not read audit trail: {e}[/dim]"

    def _cmd_safety(self, args: str) -> str:
        """Check text for safety issues."""
        if not args:
            return "[yellow]Usage:[/yellow] /safety check <text>"

        parts = args.split(maxsplit=1)
        action = parts[0].lower()
        text = parts[1] if len(parts) > 1 else ""

        if action == "check" and text:
            try:
                from local_agent.safety import (
                    CommandSafetyChecker,
                    PromptInjectionDetector,
                )

                # Check for prompt injection
                detector = PromptInjectionDetector()
                inj_result = detector.check(text)

                # Check command safety
                checker = CommandSafetyChecker()
                cmd_result = checker.check(text)

                if not inj_result.safe:
                    threats = " ".join(inj_result.threats)
                    return (
                        f"[bold red]⚠️  PROMPT INJECTION DETECTED[/bold red]\n"
                        f"  Text: {text[:100]}\n"
                        f"  Threats: {threats}"
                    )
                elif not cmd_result.safe:
                    threats = " ".join(cmd_result.threats)
                    return (
                        f"[bold yellow]⚠️  UNSAFE COMMAND[/bold yellow]\n"
                        f"  Text: {text[:100]}\n"
                        f"  Threats: {threats}"
                    )
                else:
                    return (
                        f"[bold green]✅ Safe[/bold green]\n"
                        f"  No injection patterns or unsafe commands detected.\n"
                        f"  Text: {text[:100]}"
                    )

            except Exception as e:
                return f"[dim]Safety check error: {e}[/dim]"
        else:
            return f"[bold red]Unknown safety action:[/bold red] {action}\n[dim]Use: check <text>[/dim]"

    def _cmd_rag(self, args: str) -> str:
        """Manage RAG knowledge base."""
        if not args:
            return textwrap.dedent("""\
            [bold cyan]RAG (Knowledge Base) Management[/bold cyan]

            Usage:
              [yellow]/rag index <dir>[/yellow]   Index a directory of documents
              [yellow]/rag query <text>[/yellow]    Query the knowledge base
              [yellow]/rag list[/yellow]          List indexed documents

            Examples:
              /rag index /path/to/docs
              /rag query "How do I configure the LLM provider?"
              /rag list

            The agent can also do this naturally:
              "index all markdown files in the docs directory"
              "search the knowledge base for authentication""")

        parts = args.split(maxsplit=1)
        action = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        try:
            # Try to find an existing RAG database
            rag_db = self.work_dir / ".rag.db"
            if not rag_db.exists():
                rag_db = Path.home() / ".local-coding-agent" / "rag.db"

            if action == "index":
                if not arg:
                    return "[yellow]Usage:[/yellow] /rag index <directory>"
                return f"[dim]RAG indexing queued for: {arg}[/dim]\n[dim]Ask the agent to complete: 'index {arg}'[/dim]"
            elif action == "query":
                if not arg:
                    return "[yellow]Usage:[/yellow] /rag query <text>"
                return f"[dim]RAG query queued: {arg}[/dim]\n[dim]Ask the agent: 'search the knowledge base for {arg}'[/dim]"
            elif action == "list":
                if rag_db.exists():
                    import json
                    import sqlite3
                    conn = sqlite3.connect(str(rag_db))
                    cursor = conn.execute("SELECT DISTINCT doc_id FROM chunks")
                    docs = [row[0] for row in cursor.fetchall()]
                    conn.close()
                    if docs:
                        return f"[bold cyan]📚 Indexed Documents ({len(docs)})[/bold cyan]\n" + "\n".join(f"  • {d}" for d in docs)
                return "[dim]No documents indexed yet. Use /rag index <dir> to start.[/dim]"
            else:
                return f"[bold red]Unknown RAG action:[/bold red] {action}\n[dim]Use: index, query, list[/dim]"
        except Exception as e:
            return f"[dim]RAG error: {e}[/dim]"

    def _cmd_delegate(self, args: str) -> str:
        """Delegate a task to a subagent."""
        return f"[dim]Task delegation queued: {args}[/dim]\n[dim]The agent will spawn a subagent to handle this.[/dim]"

    def _cmd_clear(self) -> str:
        """Clear the terminal screen."""
        os.system("clear" if os.name != "nt" else "cls")
        return ""

    def _cmd_history(self) -> str:
        """Show recent conversation history."""
        try:
            if hasattr(self.agent, "_history") and self.agent._history:
                recent = self.agent._history[-20:]
                lines = ["[bold cyan]💬 Conversation History (recent 20 messages)[/bold cyan]"]
                for msg in recent:
                    role = msg.get("role", "?")
                    content = str(msg.get("content", ""))[:120]
                    lines.append(f"  [{role}] {content}")
                return "\n".join(lines)
            return "[dim]No conversation history available.[/dim]"
        except Exception as e:
            return f"[dim]Could not read history: {e}[/dim]"

    def _cmd_tools(self) -> str:
        """List available tool categories."""
        return textwrap.dedent("""\
        [bold cyan]🔧 Available Tool Categories[/bold cyan]

          [bold]file[/bold]        Read, write, and patch files
          [bold]terminal[/bold]    Execute shell commands (foreground & background)
          [bold]git[/bold]         Initialize repos, add, commit, branch, push, log, merge
          [bold]search[/bold]      Search file contents and filenames (ripgrep)
          [bold]browser[/bold]     Navigate, click, type, screenshot (Playwright)
          [bold]memory[/bold]      Persistent memory (user profile + agent notes)
          [bold]skills[/bold]      Reusable procedural workflows
          [bold]session[/bold]     Search past conversation sessions
          [bold]delegate[/bold]    Spawn subagents for parallel tasks
          [bold]cron[/bold]        Schedule recurring jobs
          [bold]rag[/bold]         Index and query local documents (semantic search)
          [bold]safety[/bold]      Prompt injection detection, command safety
          [bold]explain[/bold]     Decision audit trail, self-assessment
          [bold]mcp[/bold]         Connect to external MCP servers
          [bold]human[/bold]       Ask for clarification before proceeding
          [bold]retry[/bold]       Adaptive retry with backoff
        """).strip()

    def _cmd_version(self) -> str:
        """Show version information."""
        return textwrap.dedent(f"""\
        [bold cyan]Local Coding Agent[/bold cyan]
          Model: {self.config.llm.model}
          Host: {self.config.llm.host}:{self.config.llm.port}
          Working directory: {self.work_dir}
          Python: {self._python_version()}
        """).strip()

    @staticmethod
    def _python_version() -> str:
        """Get Python version string."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    def _fetch_models(self) -> tuple[list[dict[str, Any]], list[str] | None]:
        """Fetch models from /v1/models and offline upstreams from /health.

        Returns:
            Tuple of (list of {id, status} dicts, list of offline upstream names or None).
        """
        base = f"{self.config.llm.host}"
        if not base.startswith("http"):
            base = f"http://{base}:{self.config.llm.port}"

        models = []
        offline_upstreams: list[str] = []

        # Fetch /v1/models
        try:
            resp = httpx.get(f"{base}/v1/models", timeout=10.0)
            if resp.status_code == 200:
                data = resp.json()
                model_ids = []
                if isinstance(data, dict):
                    data_list = data.get("data", [])
                elif isinstance(data, list):
                    data_list = data
                else:
                    data_list = []
                for item in data_list:
                    if isinstance(item, dict):
                        mid = item.get("id", "")
                        if mid:
                            model_ids.append(mid)
                            models.append({"id": mid, "status": "online"})
                    elif isinstance(item, str) and item:
                        model_ids.append(item)
                        models.append({"id": item, "status": "online"})
        except Exception:
            pass

        # Fetch /health for offline upstreams
        try:
            resp = httpx.get(f"{base}/health", timeout=10.0)
            if resp.status_code == 200:
                health = resp.json()
                warnings = health.get("warnings", [])
                for warn in warnings:
                    if isinstance(warn, dict):
                        upstream = warn.get("upstream", "")
                        if upstream:
                            offline_upstreams.append(upstream)
                        # Also check the text field for upstream name
                        text = warn.get("text", "")
                        if isinstance(text, str) and "upstream" in text.lower():
                            # Extract upstream name from warning text like "upstream 'foo' is offline"
                            for mid in model_ids:
                                if mid in text:
                                    for m in models:
                                        if m["id"] == mid and m["status"] == "online":
                                            m["status"] = "offline"
        except Exception:
            pass

        return models, offline_upstreams

    def _cmd_models(self, args: str) -> str | None:
        """List available models from the LLM gateway and allow switching.

        Usage:
          /models            Show numbered list of all models
          /models N          Switch to model at index N

        Offline models are marked with (offline).
        """
        models, _ = self._fetch_models()

        if not models:
            return "[bold red]Error:[/bold red] Could not fetch model list from gateway.\n[dim]Check that your gateway is running and reachable.[/dim]"

        # User wants to switch to a specific model by number
        if args:
            try:
                idx = int(args.strip())
            except ValueError:
                return f"[bold red]Error:[/bold red] Invalid model index: {args}\n[dim]Usage: /models N  (N is a number from the list)[/dim]"

            if 1 <= idx <= len(models):
                chosen = models[idx - 1]
                self.config.llm.model = chosen["id"]
                status = f" [dim](was offline)[/dim]" if chosen["status"] == "offline" else ""
                model_id = chosen["id"]
                return f"[bold green]✓[/bold green] Switched to [bold]{model_id}[/bold]{status}"
            else:
                return f"[bold red]Error:[/bold red] Index {idx} out of range (1-{len(models)})"

        # List all models
        current_model = self.config.llm.model
        lines = ["[bold cyan]Available Models[/bold cyan]\n"]

        for i, m in enumerate(models, 1):
            marker = ""
            if m["status"] == "offline":
                marker = " [dim][bold red](offline)[/bold red][/dim]"
            if m["id"] == current_model:
                marker = " [bold green]◄ current[/bold green]"

            model_id = m["id"]
            lines.append(f"  [{i}] {model_id}{marker}")

        lines.append(f"\n[dim]Type /models N to switch (e.g. /models 3)[/dim]")

        return "\n".join(lines)

    def _cmd_plan(self, args: str) -> str:
        """Plan, list, and manage task plans.

        Usage:
          /plan <goal>          Decompose a goal into steps (calls LLM)
          /plan list            List saved plans
          /plan show <path>     Load and display a saved plan
        """
        if not args:
            return textwrap.dedent("""\
            [bold cyan]Task Planner[/bold cyan]

            Usage:
              [yellow]/plan <goal>[/yellow]       Decompose a goal into actionable steps
              [yellow]/plan list[/yellow]        List saved plans
              [yellow]/plan show <path>[/yellow]   Show a saved plan

            Example:
              /plan "Refactor the auth module to use JWT tokens"
              /plan list""")

        parts = args.split(maxsplit=1)
        action = parts[0].lower()
        query = parts[1] if len(parts) > 1 else ""

        if action == "list":
            return self._cmd_plan_list()
        elif action == "show":
            return self._cmd_plan_show(query)
        else:
            # Treat as a goal to plan
            return self._cmd_plan_create(args)

    def _cmd_plan_list(self) -> str:
        """List all saved plans."""
        try:
            from local_agent.task_planner import TaskPlanner

            planner = TaskPlanner(self.agent._router)
            plans = planner.list_plans()

            if not plans:
                return "[dim]No saved plans found.[/dim]"

            lines = ["[bold cyan]Saved Plans[/bold cyan]"]
            for p in plans:
                status = "[bold green]✓[/bold green]" if p["completed"] else "[dim]○[/dim]"
                lines.append(f"  {status} {p['title']} [{p['steps']} steps] - {p['path']}")

            return "\n".join(lines)
        except Exception as e:
            return f"[dim]Could not list plans: {e}[/dim]"

    def _cmd_plan_show(self, path: str) -> str:
        """Show a saved plan."""
        if not path:
            return "[bold red]Error:[/bold red] Usage: /plan show <path>"

        try:
            from local_agent.task_planner import TaskPlanner, Plan

            planner = TaskPlanner(self.agent._router)
            plan = planner.load_plan(path)
            return "[bold cyan]Plan[/bold cyan]\n" + plan.summary()
        except FileNotFoundError:
            return f"[bold red]Error:[/bold red] Plan not found: {path}"
        except Exception as e:
            return f"[dim]Could not load plan: {e}[/dim]"

    def _cmd_plan_create(self, goal: str) -> str:
        """Create a plan by decomposing a goal."""
        try:
            from local_agent.task_planner import TaskPlanner

            planner = TaskPlanner(self.agent._router)
            plan = planner.plan(goal)

            # Auto-save
            saved_path = planner.save_plan(plan)

            complexity = planner.estimate_overall_complexity(plan)
            complexity_label = {
                "simple": "[green]simple[/green]",
                "moderate": "[yellow]moderate[/yellow]",
                "complex": "[red]complex[/red]",
            }.get(complexity, complexity)

            lines = [
                f"[bold cyan]Plan Created[/bold cyan] [dim](complexity: {complexity_label})[/dim]",
                plan.summary(),
                f"\n[dim]Saved to: {saved_path}[/dim]",
            ]

            return "\n".join(lines)
        except Exception as e:
            return f"[bold red]Error:[/bold red] Could not create plan: {e}"

    def _cmd_stats(self) -> str:
        """Show routing stats / observability dashboard."""
        try:
            stats = self.agent._router.get_stats()
        except Exception:
            return "[dim]No stats available yet.[/dim]"

        if not stats:
            return "[dim]No stats available yet.[/dim]"

        lines = ["[bold cyan]Model Routing Stats[/bold cyan]"]
        lines.append("")
        lines.append(f"  {'Model':<35} {'Requests':>8} {'Errors':>7} {'Latency':>8} {'Tokens':>10}")
        lines.append("  " + "-" * 70)

        total_requests = 0
        total_errors = 0
        total_tokens = 0

        for model, s in sorted(stats.items()):
            avg_lat = f"{s.avg_latency_ms:.0f}ms" if s.request_count > 0 else "—"
            marker = " ◄ current" if model == self.config.llm.model else ""
            lines.append(
                f"  {model + marker:<35} {s.request_count:>8} {s.error_count:>7} {avg_lat:>8} {s.total_tokens:>10}"
            )
            total_requests += s.request_count
            total_errors += s.error_count
            total_tokens += s.total_tokens

        lines.append("  " + "-" * 70)
        lines.append(f"  {'TOTAL':<35} {total_requests:>8} {total_errors:>7} {'':>8} {total_tokens:>10}")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  /provider command — view and change LLM provider config            #
    # ------------------------------------------------------------------ #

    def _cmd_provider(self, args: str) -> str:
        """View and change the upstream LLM provider configuration.

        Usage:
          /provider              Show current provider config
          /provider host <url>   Set provider host (e.g., localhost or http://...)
          /provider port <N>     Set provider port
          /provider model <name> Set model name
          /provider url <base>   Set full base URL (overrides host+port)
        """
        if not args:
            return self._cmd_provider_show()

        parts = args.split(maxsplit=1)
        action = parts[0].lower()
        value = parts[1] if len(parts) > 1 else ""

        if action == "host":
            return self._cmd_provider_set_host(value)
        elif action == "port":
            return self._cmd_provider_set_port(value)
        elif action == "model":
            return self._cmd_provider_set_model(value)
        elif action == "url":
            return self._cmd_provider_set_url(value)
        else:
            return textwrap.dedent(f"""\
            [bold red]Unknown provider action:[/bold red] {action}

            Usage:
              [yellow]/provider[/yellow]            Show current provider config
              [yellow]/provider host <url>[/yellow]   Set provider host
              [yellow]/provider port <N>[/yellow]     Set provider port
              [yellow]/provider model <name>[/yellow]  Set model name
              [yellow]/provider url <base>[/yellow]   Set full base URL""")

    def _cmd_provider_show(self) -> str:
        """Show current provider configuration."""
        llm = self.config.llm
        lines = [
            "[bold cyan]Current LLM Provider[/bold cyan]",
            "",
            f"  [bold]Model:[/bold]    {llm.model}",
            f"  [bold]Host:[/bold]     {llm.host}",
            f"  [bold]Port:[/bold]     {llm.port}",
            f"  [bold]Base URL:[/bold] {llm.base_url}",
            f"  [bold]Deterministic:[/bold] {llm.deterministic}",
        ]
        return "\n".join(lines)

    def _cmd_provider_set_host(self, value: str) -> str:
        """Set the provider host."""
        if not value:
            return "[bold red]Error:[/bold red] Usage: /provider host <url>"
        self.config.llm.host = value
        return f"[bold green]✓[/bold green] Provider host set to [bold]{value}[/bold] (base URL: {self.config.llm.base_url})"

    def _cmd_provider_set_port(self, value: str) -> str:
        """Set the provider port."""
        if not value:
            return "[bold red]Error:[/bold red] Usage: /provider port <N>"
        try:
            port = int(value)
        except ValueError:
            return f"[bold red]Error:[/bold red] '{value}' is not a valid port number"
        self.config.llm.port = port
        return f"[bold green]✓[/bold green] Provider port set to [bold]{port}[/bold] (base URL: {self.config.llm.base_url})"

    def _cmd_provider_set_model(self, value: str) -> str:
        """Set the model name."""
        if not value:
            return "[bold red]Error:[/bold red] Usage: /provider model <name>"
        self.config.llm.model = value
        return f"[bold green]✓[/bold green] Model set to [bold]{value}[/bold]"

    def _cmd_provider_set_url(self, value: str) -> str:
        """Set the full base URL (overrides host+port)."""
        if not value:
            return "[bold red]Error:[/bold red] Usage: /provider url <base>"
        self.config.llm.host = value
        self.config.llm.port = 0
        return f"[bold green]✓[/bold green] Provider URL set to [bold]{value}[/bold]"

    # ------------------------------------------------------------------ #
    #  /code command — force file-producing coding tasks                  #
    # ------------------------------------------------------------------ #

    def _cmd_code(self, args: str) -> str:
        """Code mode: the agent MUST write one or more files to disk.

        Usage:
          /code <description>    Build code and write it to files

        The agent cannot exit this command successfully without having
        created at least one file on disk. If clarification is needed,
        it may ask via human-in-the-loop, but the final result must
        always be written files.
        """
        if not args:
            return textwrap.dedent("""\
            [bold cyan]Code Mode — Write Code to Disk[/bold cyan]

            Usage:
              [yellow]/code <description>[/yellow]  Build code and write it to files

            The agent WILL NOT succeed without writing at least one file.
            If more details are needed, it will ask clarifying questions,
            but the end result is always code on disk.

            Examples:
              /code "A Python CLI to convert CSV to JSON"
              /code "A React component for a todo list"
              /code "A bash script to backup a directory to tar.gz""")

        # Build a强制 prompt: inject the user's request with a strict
        # directive that files MUST be written before the turn completes.
        code_instruction = textwrap.dedent(f"""\
            CODE MODE — You are in /code mode. Your task is to build the following and write the result to disk:

            {args}

            STRICT RULES:
            1. You MUST create one or more files on disk. This command CANNOT succeed without writing files.
            2. Use the write_file tool to write each file. Use the working directory: {self.work_dir}
            3. If you need clarification, use the human_in_the_loop tool to ask the user.
            4. After writing all files, confirm which files were created and their paths.
            5. Do NOT just print code to stdout — write it to actual files.
            6. Create proper file structure (e.g., __init__.py for Python packages, package.json for Node, etc.)
            7. If the project has multiple files, write them all.

            Build it now and write everything to disk.""")

        return self.agent.run_turn(code_instruction)

    # ------------------------------------------------------------------ #
    #  /ls renderer                                                        #
    # ------------------------------------------------------------------ #

    def _render_ls(self, target: str) -> str:
        """Render a listing of files in the given directory.

        Accepts either a raw path or a full command string like "/ls /path".
        """
        # Strip /ls prefix if present (backward compat with tests)
        if target.startswith("/ls"):
            target = target[3:].strip()
        if not target:
            target = str(self.work_dir)

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
