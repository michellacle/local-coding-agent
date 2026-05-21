"""Agent Core — main loop that takes input, calls LLM, parses and executes tool calls."""

import json
from typing import Generator

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

    def __init__(self, router: ModelRouter, registry: ToolRegistry, streaming: bool = False):
        self._router = router
        self._registry = registry
        self._streaming = streaming
        self._history: list[dict] = []

    def run_turn(self, user_input: str) -> str:
        """Run a single agent turn.

        Args:
            user_input: The user's message or command.

        Returns:
            The agent's response text (after any tool execution).
        """
        messages = self._build_messages(user_input)

        if self._streaming:
            full_response = "".join(self._router.stream_message(messages))
        else:
            full_response = self._router.send_message(messages)

        # Try to parse tool calls from the response
        tool_result = self._try_execute_tool(full_response)
        if tool_result is not None:
            return tool_result

        return full_response

    def _build_messages(self, user_input: str) -> list[dict]:
        """Build the message list including system prompt and history.

        Args:
            user_input: The user's latest input.

        Returns:
            List of message dicts ready to send to the LLM.
        """
        system_prompt = self._build_system_prompt()
        messages: list[dict] = [
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
        base = (
            "You are a coding assistant. You can use tools to help the user. "
            "When you need to use a tool, respond with a JSON object containing "
            "\"tool\" (the tool name) and \"args\" (a dict of arguments). "
            "Example: {\"tool\": \"read_file\", \"args\": {\"path\": \"file.txt\"}}\n\n"
        )
        tools_section = self._registry.get_definitions()
        return base + tools_section

    def _try_execute_tool(self, response: str) -> str | None:
        """Try to parse and execute a tool call from the LLM response.

        Args:
            response: Raw LLM response text.

        Returns:
            The tool execution result string, or None if no tool call was found.
        """
        try:
            data = json.loads(response.strip())
        except (json.JSONDecodeError, ValueError):
            # Not JSON — treat as plain text response
            return None

        if not isinstance(data, dict) or "tool" not in data:
            return None

        tool_name = data["tool"]
        args = data.get("args", {})

        try:
            result = self._registry.execute(tool_name, args)
        except KeyError:
            return f"Error: tool '{tool_name}' not found."
        except ValueError as e:
            return f"Error executing '{tool_name}': {e}"

        # Format the result for the user
        result_str = str(result)
        if isinstance(result, dict):
            result_str = json.dumps(result, indent=2)

        # Store the exchange in history for context
        self._history.append({"role": "assistant", "content": response})
        self._history.append({"role": "user", "content": f"Tool result: {result_str}"})

        # Ask the LLM to interpret the tool result
        follow_up_messages = self._build_messages("")
        if self._streaming:
            follow_up = "".join(self._router.stream_message(follow_up_messages))
        else:
            follow_up = self._router.send_message(follow_up_messages)

        return follow_up
