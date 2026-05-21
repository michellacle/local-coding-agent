"""Agent Core — main loop that takes input, calls LLM, parses and executes tool calls."""

from __future__ import annotations

import json
import re
from typing import Any

from local_agent.model_router import ModelRouter
from local_agent.tool_registry import ToolRegistry


class AgentCore:
    """Main agent loop.

    Takes user input, sends it to the LLM (with tool definitions in the
    system prompt), parses any tool calls from the response, executes them,
    and returns the result.

    Dependencies are injected for testability:
        - ModelRouter: handles all LLM communication
        - ToolRegistry: handles tool registration and execution
    """

    def __init__(
        self, router: ModelRouter, registry: ToolRegistry, streaming: bool = False
    ) -> None:
        self._router = router
        self._registry = registry
        self._streaming = streaming
        self._history: list[dict[str, Any]] = []

    def run_turn(self, user_input: str) -> str:
        """Run a single agent turn.

        Args:
            user_input: The user's message or command.

        Returns:
            The agent's response text (after any tool execution).
        """
        messages = self._build_messages(user_input)

        if self._streaming:
            full_response: str = "".join(self._router.stream_message(messages))
        else:
            full_response = self._router.send_message(messages)

        # Try to parse tool calls from the response
        tool_result: str | None = self._try_execute_tool(full_response)
        if tool_result is not None:
            return tool_result

        return full_response

    def _build_messages(self, user_input: str) -> list[dict[str, Any]]:
        """Build the message list including system prompt and history.

        Args:
            user_input: The user's latest input.

        Returns:
            List of message dicts ready to send to the LLM.
        """
        system_prompt: str = self._build_system_prompt()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        messages.extend(self._history)
        messages.append({"role": "user", "content": user_input})
        return messages

    def _build_system_prompt(self) -> str:
        """Build the system prompt with tool definitions.

        Returns:
            The full system prompt string.
        """
        base: str = (
            "You are a coding assistant. You can use tools to help the user.\n\n"
            "When you need to use a tool, respond with ONLY a JSON object containing "
            "\"tool\" (the tool name) and \"args\" (a dict of arguments). Do NOT include "
            "any text before or after the JSON.\n\n"
            'Example: {"tool": "read_file", "args": {"path": "file.txt"}}\n\n'
        )
        tools_section: str = self._registry.get_definitions()
        return base + tools_section

    def _try_execute_tool(self, response: str) -> str | None:
        """Try to parse and execute a tool call from the LLM response.

        Args:
            response: Raw LLM response text.

        Returns:
            The tool execution result string, or None if no tool call was found.
        """
        # First try parsing the whole response as JSON
        data: Any | None = None
        try:
            data = json.loads(response.strip())
        except (json.JSONDecodeError, ValueError):
            # If that fails, try to extract a JSON object from the response
            # Pattern handles one level of nested braces (e.g., "args": {"key": "val"})
            match = re.search(r'\{[^{}]*"tool"\s*:[^{}]*\{[^{}]*\}[^{}]*\}', response, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except (json.JSONDecodeError, ValueError):
                    pass

        if data is None or not isinstance(data, dict) or "tool" not in data:
            return None

        tool_name: str = data["tool"]
        args: dict[str, Any] = data.get("args", {})

        try:
            result: Any = self._registry.execute(tool_name, args)
        except KeyError:
            return f"Error: tool '{tool_name}' not found."
        except ValueError as e:
            return f"Error executing '{tool_name}': {e}"

        # Format the result for the user
        result_str: str = str(result)
        if isinstance(result, dict):
            result_str = json.dumps(result, indent=2)

        # Store the exchange in history for context
        self._history.append({"role": "assistant", "content": response})
        self._history.append({"role": "user", "content": f"Tool result: {result_str}"})

        # Ask the LLM to interpret the tool result
        follow_up_messages: list[dict[str, Any]] = self._build_messages("")
        if self._streaming:
            follow_up: str = "".join(self._router.stream_message(follow_up_messages))
        else:
            follow_up = self._router.send_message(follow_up_messages)

        return follow_up
