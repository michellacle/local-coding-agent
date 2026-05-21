"""Agent Core — main loop that takes input, calls LLM, parses and executes tool calls.

Supports multi-turn tool chaining: after each tool execution, the result is fed
back to the LLM and the loop continues until the LLM returns a plain-text
response or max_turns is reached.
"""

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
    and loops back with the result — chaining tool calls until the LLM
    returns a final plain-text answer.

    Dependencies are injected for testability:
        - ModelRouter: handles all LLM communication
        - ToolRegistry: handles tool registration and execution
    """

    def __init__(
        self,
        router: ModelRouter,
        registry: ToolRegistry,
        streaming: bool = False,
        max_turns: int = 10,
    ) -> None:
        self._router = router
        self._registry = registry
        self._streaming = streaming
        self._max_turns = max_turns
        self._history: list[dict[str, Any]] = []

    def run_turn(self, user_input: str) -> str:
        """Run a single agent turn with multi-turn tool chaining.

        Sends the user input to the LLM. If the LLM returns a tool call,
        executes it, appends the result to history, and asks the LLM again.
        Repeats until the LLM returns plain text or max_turns is reached.

        Args:
            user_input: The user's message or command.

        Returns:
            The agent's final response text.
        """
        # Append user message to history
        self._history.append({"role": "user", "content": user_input})

        final_response: str = ""

        for turn_num in range(self._max_turns):
            messages = self._build_messages()

            if self._streaming:
                full_response: str = "".join(self._router.stream_message(messages))
            else:
                full_response = self._router.send_message(messages)

            # Store assistant response in history
            self._history.append({"role": "assistant", "content": full_response})

            # Try to parse and execute a tool call
            tool_result = self._try_execute_tool(full_response)
            if tool_result is not None:
                # Tool was executed — feed result back and continue looping
                self._history.append({"role": "user", "content": f"Tool result: {tool_result}"})
                final_response = tool_result  # track last result for debugging
                continue

            # No tool call — this is the final plain-text response
            return full_response

        # Max turns reached — return last response
        return final_response if final_response else self._last_assistant_response

    @property
    def _last_assistant_response(self) -> str:
        """Get the last assistant message from history, or empty string."""
        for msg in reversed(self._history):
            if msg["role"] == "assistant":
                return msg["content"]
        return ""

    def _build_messages(self) -> list[dict[str, Any]]:
        """Build the message list including system prompt and history.

        Returns:
            List of message dicts ready to send to the LLM.
        """
        system_prompt: str = self._build_system_prompt()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        messages.extend(self._history)
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
            "any text before or after the JSON. One tool call at a time — the result will "
            "be fed back to you so you can make the next call.\n\n"
            "When you have completed the task and have a final answer for the user, "
            "respond with plain text (no JSON).\n\n"
            'Example tool call: {"tool": "read_file", "args": {"path": "file.txt"}}\n\n'
            'Example final answer: "I have created the file successfully."\n\n'
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
            match = re.search(
                r'\{[^{}]*"tool"\s*:[^{}]*\{[^{}]*\}[^{}]*\}',
                response,
                re.DOTALL,
            )
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

        # Format the result for feeding back to the LLM
        result_str: str = str(result)
        if isinstance(result, dict):
            result_str = json.dumps(result, indent=2)

        return result_str

    def reset_history(self) -> None:
        """Clear conversation history. Call between independent user turns."""
        self._history.clear()
