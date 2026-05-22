"""Model Router — LLM client with streaming, complexity routing, and fallback chains.

Routes requests to the appropriate model based on task complexity, and falls
back to alternate providers on failures.

Works with Ollama, vLLM, and any server exposing /v1/chat/completions.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Generator

import httpx

from local_agent.config import LLMConfig


# Complexity tiers for model routing
COMPLEXITY_SIMPLE = "simple"
COMPLEXITY_MODERATE = "moderate"
COMPLEXITY_COMPLEX = "complex"


@dataclass
class RoutingRule:
    """Routing rule: map complexity to a model config.

    Attributes:
        complexity: Which complexity tier this rule applies to.
        config: LLMConfig for this tier.
    """
    complexity: str
    config: LLMConfig


@dataclass
class RoutingStats:
    """Token usage and latency tracking per model."""

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    request_count: int = 0
    error_count: int = 0

    @property
    def avg_latency_ms(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.latency_ms / self.request_count


class ModelRouter:
    """Client for sending messages to an LLM via OpenAI-compatible API.

    Supports:
    - Single-provider mode (backward compatible — just pass one LLMConfig)
    - Complexity-based routing (simple/moderate/complex -> different models)
    - Fallback chains (try alternate providers on failure)
    - Token usage tracking and latency stats
    - Streaming responses
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        routing_rules: list[RoutingRule] | None = None,
        fallbacks: list[LLMConfig] | None = None,
    ) -> None:
        """Initialize the model router.

        Args:
            config: Primary LLMConfig. Required if routing_rules is None.
            routing_rules: Optional rules mapping complexity to configs.
                Overrides config when set.
            fallbacks: Optional list of LLMConfigs to try on failure.
        """
        if config is None and routing_rules is None:
            raise ValueError("ModelRouter requires config or routing_rules")

        # Single-config mode (backward compatible)
        if config is not None:
            self.config = config
        else:
            # Use the first rule's config as the default
            self.config = routing_rules[0].config

        self.routing_rules = routing_rules or []
        self.fallbacks = fallbacks or []
        self._stats: dict[str, RoutingStats] = {}

    def _get_stats(self, model: str) -> RoutingStats:
        """Get or create routing stats for a model."""
        if model not in self._stats:
            self._stats[model] = RoutingStats(model=model)
        return self._stats[model]

    def record_request(self, model: str, latency_ms: float, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Record a successful request for stats tracking.

        Args:
            model: Model name.
            latency_ms: Request latency in milliseconds.
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.
        """
        stats = self._get_stats(model)
        stats.request_count += 1
        stats.latency_ms += latency_ms
        stats.prompt_tokens += prompt_tokens
        stats.completion_tokens += completion_tokens
        stats.total_tokens += prompt_tokens + completion_tokens

    def record_error(self, model: str) -> None:
        """Record a failed request for stats tracking."""
        stats = self._get_stats(model)
        stats.error_count += 1

    def get_stats(self, model: str | None = None) -> dict[str, RoutingStats]:
        """Get routing stats.

        Args:
            model: If provided, return stats for that model only.

        Returns:
            Dict mapping model name to RoutingStats.
        """
        if model:
            return {model: self._stats.get(model, RoutingStats(model=model))}
        # Return stats for all known models, including fallbacks
        all_models = set(self._stats.keys())
        all_models.add(self.config.model)
        for rule in self.routing_rules:
            all_models.add(rule.config.model)
        for fb in self.fallbacks:
            all_models.add(fb.model)
        return {m: self._stats.get(m, RoutingStats(model=m)) for m in all_models}

    def get_model_for_complexity(self, complexity: str = COMPLEXITY_MODERATE) -> LLMConfig:
        """Select the best model config for a given complexity tier.

        Args:
            complexity: One of 'simple', 'moderate', 'complex'.

        Returns:
            LLMConfig for the selected model.
        """
        if self.routing_rules:
            for rule in self.routing_rules:
                if rule.complexity == complexity:
                    return rule.config
        # Default: use primary config
        return self.config

    def get_routing_chain(self, complexity: str = COMPLEXITY_MODERATE) -> list[LLMConfig]:
        """Get the ordered list of configs to try: primary + fallbacks.

        Args:
            complexity: Complexity tier for primary model selection.

        Returns:
            Ordered list of LLMConfigs to attempt.
        """
        primary = self.get_model_for_complexity(complexity)
        chain = [primary]
        for fb in self.fallbacks:
            if fb.model != primary.model:
                chain.append(fb)
        return chain

    def send_message(
        self,
        messages: list[dict[str, Any]],
        complexity: str | None = None,
    ) -> str:
        """Send messages with optional complexity routing and fallback.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            complexity: Optional complexity tier. If set, routes to the
                appropriate model and tries fallbacks on failure.

        Returns:
            The assistant's response text.

        Raises:
            Exception: If all providers in the chain fail.
        """
        if complexity:
            chain = self.get_routing_chain(complexity)
        else:
            chain = [self.config]
            for fb in self.fallbacks:
                if fb.model != self.config.model:
                    chain.append(fb)

        last_error: Exception | None = None

        for cfg in chain:
            try:
                return self._send_to(cfg, messages)
            except Exception as e:
                last_error = e
                self.record_error(cfg.model)

        # All failed
        raise last_error  # type: ignore[misc]

    def _send_to(self, config: LLMConfig, messages: list[dict[str, Any]]) -> str:
        """Send to a specific LLM config.

        Args:
            config: LLMConfig to use.
            messages: Messages to send.

        Returns:
            Response text.
        """
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
        }

        if config.deterministic:
            payload["temperature"] = 0
            payload["seed"] = 0

        url = f"{config.base_url}/chat/completions"
        start = time.time()

        resp = httpx.post(url, json=payload, timeout=120.0)
        resp.raise_for_status()

        latency_ms = (time.time() - start) * 1000

        data: Any = resp.json()
        content: str = data["choices"][0]["message"]["content"]

        # Track tokens from response if available
        prompt_tokens = 0
        completion_tokens = 0
        if "usage" in data:
            prompt_tokens = data["usage"].get("prompt_tokens", 0)
            completion_tokens = data["usage"].get("completion_tokens", 0)

        self.record_request(config.model, latency_ms, prompt_tokens, completion_tokens)
        return content

    def stream_message(
        self,
        messages: list[dict[str, Any]],
        complexity: str | None = None,
    ) -> Generator[str, None, None]:
        """Stream messages with optional complexity routing.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            complexity: Optional complexity tier for model selection.

        Yields:
            Non-empty content strings from the streaming response.
        """
        if complexity:
            config = self.get_model_for_complexity(complexity)
        else:
            config = self.config

        yield from self._stream_to(config, messages)

    def _stream_to(
        self,
        config: LLMConfig,
        messages: list[dict[str, Any]],
    ) -> Generator[str, None, None]:
        """Stream from a specific LLM config.

        Args:
            config: LLMConfig to use.
            messages: Messages to send.

        Yields:
            Non-empty content strings.
        """
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "stream": True,
        }

        if config.deterministic:
            payload["temperature"] = 0
            payload["seed"] = 0

        url = f"{config.base_url}/chat/completions"
        start = time.time()

        with httpx.Client() as client:
            with client.stream("POST", url, json=payload, timeout=120.0) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines():
                    if isinstance(raw_line, bytes):
                        line: str = raw_line.decode("utf-8").strip()
                    else:
                        line = raw_line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str: str = line[6:]
                    if data_str == "[DONE]":
                        latency_ms = (time.time() - start) * 1000
                        self.record_request(config.model, latency_ms)
                        break
                    try:
                        chunk: dict[str, Any] = json.loads(data_str)
                        content: str = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
