"""Main entry point for the local coding agent."""

from __future__ import annotations

import argparse
import sys
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

    # Build agent
    agent: AgentCore = AgentCore(
        router=router, registry=registry, streaming=False, max_turns=args.max_turns
    )

    # Non-interactive mode
    if args.prompt is not None:
        _run_non_interactive(agent, args.prompt)
        return

    ui: TerminalUI = TerminalUI(agent=agent, config=app_config)

    console: Console = Console()
    console.print(ui.render_welcome())

    # Main loop — reset history between independent user turns
    try:
        while True:
            raw: str = input(ui.render_input_prompt().replace("[", "").replace("]", ""))
            cleaned: str | None = ui.process_input(raw)
            if cleaned is None:
                continue
            if ui.should_stop(cleaned):
                console.print("[dim]Goodbye![/dim]")
                break

            response: str = ui.run_turn(cleaned)
            console.print(ui.render_agent_response(response))
            agent.reset_history()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted. Goodbye![/dim]")
        sys.exit(0)


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
