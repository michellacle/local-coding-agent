"""Main entry point for the local coding agent."""

from __future__ import annotations

import argparse
import os
import sys

# Enable readline: left/right cursor movement + up/down history recall
try:
    import readline
except ImportError:
    readline = None

from pathlib import Path
from rich.console import Console

from local_agent.config import LLMConfig, AppConfig
from local_agent.model_router import ModelRouter
from local_agent.tool_registry import ToolRegistry, ToolSchema
from local_agent.agent_core import AgentCore
from local_agent.terminal_ui import TerminalUI
from local_agent.tools.file_tools import read_file, write_file, patch_file
from local_agent.tools.terminal_tools import execute_command
from local_agent.tools.git_tools import git_init, git_add, git_commit, git_status, git_diff
from local_agent.tools.search_tools import search_files, list_directory
from local_agent.tools.human_loop import clarify, confirm, BlockingInteraction, ClarificationRequest, ApprovalRequest
from local_agent.tools.memory_store import memory, memory_search, memory_list
from local_agent.tools.skill_manage import skill_manage, skill_view, skills_list
from local_agent.tools.session_search import session_search
from local_agent.multi_agent import delegate_task, delegate_batch


def register_tools(registry: ToolRegistry) -> None:
    """Register all available tools with the registry."""
    registry.register(
        schema=ToolSchema(
            name="read_file",
            description="Read a text file with optional line offset and limit",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "offset": {"type": "integer", "description": "1-based starting line"},
                    "limit": {"type": "integer", "description": "Max lines to read"},
                },
                "required": ["path"],
            },
        ),
        fn=read_file,
    )

    registry.register(
        schema=ToolSchema(
            name="write_file",
            description="Write content to a file, creating parent directories if needed",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination file path"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        ),
        fn=write_file,
    )

    registry.register(
        schema=ToolSchema(
            name="patch_file",
            description="Find-and-replace text in a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File to patch"},
                    "old_string": {"type": "string", "description": "Text to find"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        ),
        fn=patch_file,
    )

    registry.register(
        schema=ToolSchema(
            name="execute_command",
            description="Execute a shell command",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds"},
                    "workdir": {"type": "string", "description": "Working directory"},
                },
                "required": ["command"],
            },
        ),
        fn=execute_command,
    )

    registry.register(
        schema=ToolSchema(
            name="git_init",
            description="Initialize a git repository",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory to initialize"},
                },
                "required": ["path"],
            },
        ),
        fn=git_init,
    )

    registry.register(
        schema=ToolSchema(
            name="git_add",
            description="Stage files for commit",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repository directory"},
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Files to stage",
                    },
                },
                "required": ["path", "files"],
            },
        ),
        fn=git_add,
    )

    registry.register(
        schema=ToolSchema(
            name="git_commit",
            description="Create a commit",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repository directory"},
                    "message": {"type": "string", "description": "Commit message"},
                },
                "required": ["path", "message"],
            },
        ),
        fn=git_commit,
    )

    registry.register(
        schema=ToolSchema(
            name="git_status",
            description="Get git status",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repository directory"},
                },
                "required": ["path"],
            },
        ),
        fn=git_status,
    )

    registry.register(
        schema=ToolSchema(
            name="git_diff",
            description="Get git diff",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repository directory"},
                },
                "required": ["path"],
            },
        ),
        fn=git_diff,
    )

    registry.register(
        schema=ToolSchema(
            name="search_files",
            description="Search for files by name pattern or content regex. Use content search to find code matching a pattern, or filename search to find files by name.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for in file contents or filenames",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default: current directory)",
                    },
                    "file_glob": {
                        "type": "string",
                        "description": "Optional glob to filter files (e.g., '*.py')",
                    },
                    "search_content": {
                        "type": "boolean",
                        "description": "If true, search file contents. If false, match filenames. Default: true.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default: 50)",
                    },
                },
                "required": ["pattern"],
            },
        ),
        fn=search_files,
    )

    registry.register(
        schema=ToolSchema(
            name="list_directory",
            description="List directory contents as a sorted tree. Shows files and subdirectories up to the specified depth.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list (default: current directory)",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Max recursion depth (default: 3)",
                    },
                    "show_hidden": {
                        "type": "boolean",
                        "description": "Include dotfiles and hidden directories (default: false)",
                    },
                    "file_glob": {
                        "type": "string",
                        "description": "Only list files matching this glob pattern (e.g., '*.py')",
                    },
                },
                "required": [],
            },
        ),
        fn=list_directory,
    )

    registry.register(
        schema=ToolSchema(
            name="clarify",
            description="Ask the user a clarification question. Supports multiple choice (up to 4 options) or open-ended questions.",
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to present to the user.",
                    },
                    "choices": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Up to 4 answer choices. Omit for open-ended questions.",
                    },
                },
                "required": ["question"],
            },
        ),
        fn=clarify,
    )

    registry.register(
        schema=ToolSchema(
            name="confirm",
            description="Request user approval before proceeding with a sensitive or destructive action.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Short description of the action to approve.",
                    },
                    "details": {
                        "type": "string",
                        "description": "Extra context about what will happen.",
                    },
                },
                "required": ["action"],
            },
        ),
        fn=confirm,
    )

    registry.register(
        schema=ToolSchema(
            name="memory",
            description="Save, update, or remove persistent memory entries. Memory survives across sessions.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "One of 'add', 'replace', 'remove'.",
                    },
                    "target": {
                        "type": "string",
                        "description": "'user' for user profile, 'memory' for agent notes.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Entry content. Required for 'add' and 'replace'.",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Substring identifying an existing entry. Required for 'replace' and 'remove'.",
                    },
                },
                "required": ["action", "target"],
            },
        ),
        fn=memory,
    )

    registry.register(
        schema=ToolSchema(
            name="memory_search",
            description="Search persistent memory entries using full-text search.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (keywords, phrases).",
                    },
                    "target": {
                        "type": "string",
                        "description": "If set, only search in this target ('user' or 'memory').",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return.",
                    },
                },
                "required": ["query"],
            },
        ),
        fn=memory_search,
    )

    registry.register(
        schema=ToolSchema(
            name="memory_list",
            description="List all memory entries, optionally filtered by target.",
            parameters={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "If set, only list entries for this target.",
                    },
                },
                "required": [],
            },
        ),
        fn=memory_list,
    )

    registry.register(
        schema=ToolSchema(
            name="skill_manage",
            description="Manage skills (create, update, delete, patch). Skills are procedural memory for reusable workflows.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "One of 'create', 'patch', 'edit', 'delete', 'write_file', 'remove_file'.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Skill name (lowercase, hyphens/underscores).",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full SKILL.md content. Required for 'create' and 'edit'.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Text to find (required for 'patch').",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text (required for 'patch').",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences (default: false).",
                    },
                    "category": {
                        "type": "string",
                        "description": "Category for organizing skills (create only).",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to a supporting file (write_file/remove_file).",
                    },
                    "file_content": {
                        "type": "string",
                        "description": "Content for write_file.",
                    },
                },
                "required": ["action", "name"],
            },
        ),
        fn=skill_manage,
    )

    registry.register(
        schema=ToolSchema(
            name="skill_view",
            description="View a skill's content or a linked file.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name.",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Optional path to a linked file.",
                    },
                },
                "required": ["name"],
            },
        ),
        fn=skill_view,
    )

    registry.register(
        schema=ToolSchema(
            name="skills_list",
            description="List available skills with descriptions.",
            parameters={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Optional category filter.",
                    },
                },
                "required": [],
            },
        ),
        fn=skills_list,
    )

    registry.register(
        schema=ToolSchema(
            name="session_search",
            description="Search past conversation sessions for context from previous discussions. Supports discovery (query), scroll (session_id + around_message_id), and browse (no args) modes.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords/phrases (discovery mode).",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session to scroll inside (scroll mode).",
                    },
                    "around_message_id": {
                        "type": "integer",
                        "description": "Message to center on (scroll mode).",
                    },
                    "window": {
                        "type": "integer",
                        "description": "Context window for scroll mode (1-20, default 5).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max sessions to return (default 3, max 10).",
                    },
                    "role_filter": {
                        "type": "string",
                        "description": "Filter by role(s) e.g., 'user,assistant'.",
                    },
                },
                "required": [],
            },
        ),
        fn=session_search,
    )

    registry.register(
        schema=ToolSchema(
            name="delegate_task",
            description="Spawn a subagent to work on a task in an isolated context. The child agent gets its own conversation, terminal session, and toolset.",
            parameters={
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "What the child agent should accomplish.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Background information the child agent needs.",
                    },
                    "toolsets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Toolsets to enable for this subagent (e.g., ['file', 'terminal']).",
                    },
                },
                "required": ["goal"],
            },
        ),
        fn=delegate_task,
    )

    registry.register(
        schema=ToolSchema(
            name="delegate_batch",
            description="Delegate multiple tasks to parallel subagents (up to 3 concurrently).",
            parameters={
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "goal": {"type": "string"},
                                "context": {"type": "string"},
                                "toolsets": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                        "description": "List of task dicts with goal, context, and toolsets.",
                    },
                },
                "required": ["tasks"],
            },
        ),
        fn=delegate_batch,
    )


def _handle_clarification(console: Console, req: ClarificationRequest) -> str:
    """Present a clarification question and collect the user's answer."""
    console.print(f"\n[bold yellow]?[/bold yellow] {req.question}")
    if req.is_multiple_choice:
        for i, choice in enumerate(req.choices, 1):
            console.print(f"  [bold]{i}[/bold]. {choice}")
        console.print(f"  [bold]{len(req.choices) + 1}[/bold]. Other (type your answer)")
        while True:
            raw = input("  > ").strip()
            if not raw:
                continue
            idx = int(raw) if raw.isdigit() else -1
            if 1 <= idx <= len(req.choices):
                return req.choices[idx - 1]
            if idx == len(req.choices) + 1:
                other = input("  Your answer: ").strip()
                return other or "No answer provided."
            console.print("  [dim]Invalid selection, try again.[/dim]")
    else:
        raw = input("  > ").strip()
        return raw or "No answer provided."


def _handle_approval(console: Console, req: ApprovalRequest) -> str:
    """Present an approval prompt and collect yes/no."""
    msg = f"[bold red]⚠[/bold red] {req.action}"
    if req.details:
        msg += f"\n    [dim]{req.details}[/dim]"
    console.print(f"\n{msg}")
    while True:
        raw = input("  Proceed? [y/N] ").strip().lower()
        if raw in ("y", "yes"):
            return "approved"
        if raw in ("n", "no", ""):
            return "denied"
        console.print("  [dim]Please answer y or n.[/dim]")


def main() -> None:
    """Run the local coding agent."""
    parser = argparse.ArgumentParser(description="Local coding agent")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to run in non-interactive mode (runs one turn with tool chaining and exits)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Max agent turns per prompt for tool chaining (default: 10)",
    )
    args = parser.parse_args()

    # Load config from environment
    llm_config: LLMConfig = LLMConfig.from_env()
    app_config: AppConfig = AppConfig(
        work_dir=Path.cwd(),
        llm=llm_config,
    )

    # Build components
    router: ModelRouter = ModelRouter(llm_config)
    registry: ToolRegistry = ToolRegistry()
    register_tools(registry)

    console: Console = Console()

    # Build human-in-the-loop callback for interactive mode
    def _human_io_handler(exc: BlockingInteraction) -> str:
        """Handle blocking human interactions in the terminal."""
        req = exc.request
        if isinstance(req, ClarificationRequest):
            return _handle_clarification(console, req)
        if isinstance(req, ApprovalRequest):
            return _handle_approval(console, req)
        return "User did not respond."

    # Build agent
    agent: AgentCore = AgentCore(
        router=router, registry=registry, streaming=False, max_turns=args.max_turns,
        human_io=_human_io_handler,
    )

    # Non-interactive mode
    if args.prompt is not None:
        _run_non_interactive(agent, args.prompt)
        return

    ui: TerminalUI = TerminalUI(agent=agent, config=app_config)

    console: Console = Console()
    console.print(ui.render_welcome())

    # Enable readline: arrow-key history recall + left/right cursor movement
    _history_file: str | None = None
    if readline is not None:
        readline.set_history_length(500)
        _history_file = str(Path.home() / ".local-coding-agent" / "input_history")
        if os.path.isfile(_history_file):
            try:
                readline.read_history_file(_history_file)
            except Exception:
                pass

    # Main loop — reset history between independent user turns
    try:
        while True:
            raw: str = console.input(ui.render_input_prompt())
            cleaned: str | None = ui.process_input(raw)
            if cleaned is None:
                continue
            # Record every non-empty command in readline history
            if readline is not None:
                readline.add_history(cleaned)
            if ui.should_stop(cleaned):
                console.print("[dim]Goodbye![/dim]")
                break

            # Handle slash commands
            if ui.is_special_command(cleaned):
                cmd_response = ui.handle_special_command(cleaned)
                if cmd_response is None:
                    console.print("[dim]Goodbye![/dim]")
                    break
                console.print(cmd_response)
                continue

            response: str = ui.run_turn(cleaned)
            console.print(ui.render_agent_response(response))
            agent.reset_history()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted. Goodbye![/dim]")
    finally:
        # Persist readline history to disk
        if readline is not None and _history_file is not None:
            try:
                Path(_history_file).parent.mkdir(parents=True, exist_ok=True)
                readline.write_history_file(_history_file)
            except Exception:
                pass


def _run_non_interactive(agent: AgentCore, prompt: str) -> None:
    """Run the agent in non-interactive mode.

    run_turn() now handles full multi-turn tool chaining internally,
    so a single call handles the entire prompt -> tool chain -> final answer.
    """
    console: Console = Console()
    console.print(f"[bold blue]>[/bold blue] {prompt}")
    response: str = agent.run_turn(prompt)
    console.print(f"[bold green]Assistant[/bold green]: {response}")


if __name__ == "__main__":
    main()
