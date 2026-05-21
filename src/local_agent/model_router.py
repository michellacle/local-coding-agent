"""Model Router — LLM client with streaming via OpenAI-compatible API."""

from __future__ import annotations

import json
from typing import Any, Generator

import httpx

from local_agent.config import LLMConfig


class ModelRouter:
    """Client for sending messages to an LLM via OpenAI-compatible API.

    Works with Ollama, vLLM, and any server exposing /v1/chat/completions.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def send_message(self, messages: list[dict[str, Any]]) -> str:
        """Send messages and return the full completion as a string.

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            The assistant's response text.
        """
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }

        if self.config.deterministic:
            payload["temperature"] = 0
            payload["seed"] = 0

        url = f"{self.config.base_url}/chat/completions"
        resp = httpx.post(url, json=payload, timeout=120.0)
        resp.raise_for_status()

        data: Any = resp.json()
        content: str = data["choices"][0]["message"]["content"]
        return content

    def stream_message(
        self, messages: list[dict[str, Any]]
    ) -> Generator[str, None, None]:
        """Send messages and yield content chunks as they arrive.

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Yields:
            Non-empty content strings from the streaming response.
        """
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
        }

        if self.config.deterministic:
            payload["temperature"] = 0
            payload["seed"] = 0

        url = f"{self.config.base_url}/chat/completions"

        with httpx.Client() as client:
            with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines():
                    # iter_lines may return bytes or str
                    if isinstance(raw_line, bytes):
                        line: str = raw_line.decode("utf-8").strip()
                    else:
                        line = raw_line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str: str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk: dict[str, Any] = json.loads(data_str)
                        content: str = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
