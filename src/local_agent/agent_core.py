"""Agent Core — main loop that takes input, calls LLM, parses and executes tool calls.

Supports multi-turn tool chaining: after each tool execution, the result is fed
back to the LLM and the loop continues until the LLM returns a plain-text
response or max_turns is reached.

Dependencies are injected for testability:
    - ModelRouter: handles all LLM communication
    - ToolRegistry: handles tool registration and execution
    - human_io: callback for human-in-the-loop interactions
    - retry_strategy: adaptive retry controller
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from local_agent.model_router import ModelRouter
from local_agent.retry import RetryStrategy
from local_agent.tool_registry import ToolRegistry
from local_agent.tools.human_loop import BlockingInteraction


HumanIoCallback = Callable[[BlockingInteraction], str]


class AgentCore:
    """Main agent loop.

    Takes user input, sends it to the LLM (with tool definitions in the
    system prompt), parses any tool calls from the response, executes them,
    and loops back with the result — chaining tool calls until the LLM
    returns a plain-text answer or max_turns is reached.

    Dependencies are injected for testability:
        - ModelRouter: handles all LLM communication
        - ToolRegistry: handles tool registration and execution
        - human_io: callback for human-in-the-loop interactions
    """

    def __init__(
        self,
        router: ModelRouter,
        registry: ToolRegistry,
        streaming: bool = False,
        max_turns: int = 10,
        human_io: HumanIoCallback | None = None,
        retry_strategy: RetryStrategy | None = None,
    ) -> None:
        self._router = router
        self._registry = registry
        self._streaming = streaming
        self._max_turns = max_turns
        self._human_io = human_io
        self._history: list[dict[str, Any]] = []
        self._retry = retry_strategy if retry_strategy is not None else RetryStrategy()

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
        self._retry.reset()
        self._history.append({"role": "user", "content": user_input})

        final_response: str = ""

        for turn_num in range(self._max_turns):
            messages = self._build_messages()

            # LLM call with retry
            full_response: str = self._call_llm_with_retry(messages)

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
            self._retry.record_success()
            return full_response

        # Max turns reached — return last response
        return final_response if final_response else self._last_assistant_response

    def _call_llm_with_retry(self, messages: list[dict[str, Any]]) -> str:
        """Call the LLM with adaptive retry on transient failures.

        Args:
            messages: Message list to send to the LLM.

        Returns:
            The LLM response string.
        """
        llm_retry = RetryStrategy(
            max_retries=self._retry.max_retries,
            max_backoff=self._retry.max_backoff,
            backoff_fn=self._retry._backoff_fn,
        )

        while True:
            try:
                if self._streaming:
                    return "".join(self._router.stream_message(messages))
                else:
                    return self._router.send_message(messages)
            except Exception as e:
                category = llm_retry.classify(e)
                llm_retry.record_failure(e)

                if not llm_retry.should_retry(e):
                    raise

                llm_retry.wait()
                llm_retry.next_attempt()

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

        # Append retry context if there are recent failures
        context = self._retry.context_summary()
        if context:
            messages.append({"role": "user", "content": context})

        return messages

    @property
    def _last_assistant_response(self) -> str:
        """Get the last assistant message from history, or empty string."""
        for msg in reversed(self._history):
            if msg["role"] == "assistant":
                return msg["content"]
        return ""

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

        # Execute with retry
        result: str = self._execute_tool_with_retry(tool_name, args)
        return result

    def _execute_tool_with_retry(self, tool_name: str, args: dict[str, Any]) -> str:
        """Execute a tool call with adaptive retry.

        Args:
            tool_name: Name of the tool to execute.
            args: Arguments to pass to the tool.

        Returns:
            The tool execution result string.
        """
        while True:
            try:
                result: Any = self._registry.execute(tool_name, args)

                # Handle human-in-the-loop
                if isinstance(result, BlockingInteraction):
                    if self._human_io is not None:
                        answer = self._human_io(result)
                        return f"User response: {answer}"
                    return "Error: blocking interaction requested but no human_io callback configured."

                # Format the result for feeding back to the LLM
                result_str: str = str(result)
                if isinstance(result, dict):
                    result_str = json.dumps(result, indent=2)

                self._retry.record_success()
                return result_str

            except BlockingInteraction as e:
                if self._human_io is not None:
                    answer = self._human_io(e)
                    return f"User response: {answer}"
                return "Error: blocking interaction requested but no human_io callback configured."

            except KeyError as e:
                # Unknown tool — permanent, don't retry
                return f"Error: tool '{tool_name}' not found."

            except ValueError as e:
                error_msg = str(e)
                category = self._retry.classify(error_msg)
                self._retry.record_failure(error_msg)

                if category == self._retry.classify("not found"):
                    return f"Error executing '{tool_name}': {e}"

                if not self._retry.should_retry(e):
                    return f"Error executing '{tool_name}': {e}"

                self._retry.next_attempt()
                continue

            except Exception as e:
                error_msg = str(e)
                category = self._retry.classify(error_msg)
                self._retry.record_failure(error_msg)

                if not self._retry.should_retry(e):
                    # Exhausted retries — try escalation
                    escalation = self._retry.escalate(error_msg)
                    if escalation is not None:
                        return escalation
                    return f"Error executing '{tool_name}': {error_msg}"

                self._retry.wait()
                self._retry.next_attempt()
                continue

    def reset_history(self) -> None:
        """Clear conversation history. Call between independent user turns."""
        self._history.clear()
        self._retry.reset()
