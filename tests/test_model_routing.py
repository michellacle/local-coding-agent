"""Tests for complexity routing and fallback chains in ModelRouter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import httpx

from local_agent.config import LLMConfig
from local_agent.model_router import (
    COMPLEXITY_COMPLEX,
    COMPLEXITY_MODERATE,
    COMPLEXITY_SIMPLE,
    ModelRouter,
    RoutingRule,
    RoutingStats,
)


# ------------------------------------------------------------------ #
# RoutingStats
# ------------------------------------------------------------------ #


class TestRoutingStats:
    def test_defaults(self):
        s = RoutingStats(model="x")
        assert s.prompt_tokens == 0
        assert s.completion_tokens == 0
        assert s.total_tokens == 0
        assert s.latency_ms == 0.0
        assert s.request_count == 0
        assert s.error_count == 0

    def test_avg_latency(self):
        s = RoutingStats(model="x")
        assert s.avg_latency_ms == 0.0

    def test_avg_latency_after_requests(self):
        s = RoutingStats(model="x")
        s.latency_ms = 300.0
        s.request_count = 3
        assert s.avg_latency_ms == 100.0


# ------------------------------------------------------------------ #
# RoutingRule
# ------------------------------------------------------------------ #


class TestRoutingRule:
    def test_creation(self):
        cfg = LLMConfig(model="small", host="localhost", port=9999)
        rule = RoutingRule(complexity="simple", config=cfg)
        assert rule.complexity == "simple"
        assert rule.config is cfg


# ------------------------------------------------------------------ #
# ModelRouter — routing + fallback
# ------------------------------------------------------------------ #


@pytest.fixture
def primary():
    return LLMConfig(model="primary", host="localhost", port=8000)


@pytest.fixture
def fallback():
    return LLMConfig(model="fallback", host="localhost", port=8001)


class TestModelRouterConstruction:
    def test_single_config(self, primary):
        router = ModelRouter(config=primary)
        assert router.config is primary
        assert router.routing_rules == []
        assert router.fallbacks == []

    def test_with_routing_rules(self, primary):
        rules = [
            RoutingRule(complexity=COMPLEXITY_SIMPLE, config=primary),
        ]
        router = ModelRouter(routing_rules=rules)
        assert router.config is primary
        assert len(router.routing_rules) == 1

    def test_with_fallbacks(self, primary, fallback):
        router = ModelRouter(config=primary, fallbacks=[fallback])
        assert router.fallbacks == [fallback]

    def test_no_config_no_rules_raises(self):
        with pytest.raises(ValueError, match="config or routing_rules"):
            ModelRouter()


class TestModelRouterRouting:
    def test_no_rules_returns_primary(self, primary):
        router = ModelRouter(config=primary)
        cfg = router.get_model_for_complexity("simple")
        assert cfg is primary

    def test_select_simple(self, primary):
        small = LLMConfig(model="small", host="localhost", port=9999)
        rules = [
            RoutingRule(complexity=COMPLEXITY_SIMPLE, config=small),
            RoutingRule(complexity=COMPLEXITY_COMPLEX, config=primary),
        ]
        router = ModelRouter(routing_rules=rules)
        cfg = router.get_model_for_complexity(COMPLEXITY_SIMPLE)
        assert cfg.model == "small"

    def test_select_complex(self, primary):
        big = LLMConfig(model="big", host="localhost", port=7777)
        rules = [
            RoutingRule(complexity=COMPLEXITY_SIMPLE, config=primary),
            RoutingRule(complexity=COMPLEXITY_COMPLEX, config=big),
        ]
        router = ModelRouter(routing_rules=rules)
        cfg = router.get_model_for_complexity(COMPLEXITY_COMPLEX)
        assert cfg.model == "big"

    def test_routing_chain_includes_fallbacks(self, primary, fallback):
        router = ModelRouter(config=primary, fallbacks=[fallback])
        chain = router.get_routing_chain()
        assert len(chain) == 2
        assert chain[0].model == "primary"
        assert chain[1].model == "fallback"

    def test_routing_chain_skips_duplicate_fallback(self, primary):
        router = ModelRouter(config=primary, fallbacks=[primary])
        chain = router.get_routing_chain()
        assert len(chain) == 1


class TestModelRouterFallback:
    def _success_resp(self):
        return MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }),
        )

    def test_fallback_on_failure(self, primary, fallback):
        with patch.object(httpx, "post") as mock_post:
            mock_post.side_effect = [
                httpx.ConnectError("primary down"),
                self._success_resp(),
            ]

            router = ModelRouter(config=primary, fallbacks=[fallback])
            result = router.send_message(
                [{"role": "user", "content": "hi"}],
                complexity=COMPLEXITY_SIMPLE,
            )
            assert result == "OK"

    def test_raises_when_all_fail(self, primary, fallback):
        with patch.object(httpx, "post") as mock_post:
            mock_post.side_effect = httpx.ConnectError("all down")

            router = ModelRouter(config=primary, fallbacks=[fallback])
            with pytest.raises(httpx.ConnectError):
                router.send_message(
                    [{"role": "user", "content": "hi"}],
                    complexity=COMPLEXITY_SIMPLE,
                )

    def test_primary_succeeds_no_fallback(self, primary, fallback):
        with patch.object(httpx, "post") as mock_post:
            mock_post.return_value = self._success_resp()

            router = ModelRouter(config=primary, fallbacks=[fallback])
            result = router.send_message([{"role": "user", "content": "hi"}])
            assert result == "OK"
            # Only one call — fallback was NOT tried
            assert mock_post.call_count == 1


class TestModelRouterStats:
    def _success_resp(self, prompt=2, completion=3):
        return MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={
                "choices": [{"message": {"content": "primary ok"}}],
                "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
            }),
        )

    def test_records_stats_on_success(self, primary):
        with patch.object(httpx, "post") as mock_post:
            mock_post.return_value = self._success_resp(prompt=10, completion=20)

            router = ModelRouter(config=primary)
            router.send_message([{"role": "user", "content": "hi"}])

        stats = router.get_stats("primary")
        assert stats["primary"].request_count == 1
        assert stats["primary"].prompt_tokens == 10
        assert stats["primary"].completion_tokens == 20
        assert stats["primary"].total_tokens == 30

    def test_records_error_on_failure(self, primary):
        fb = LLMConfig(model="fb", host="localhost", port=9999)
        with patch.object(httpx, "post") as mock_post:
            mock_post.side_effect = [
                httpx.ConnectError("fail"),
                self._success_resp(prompt=1, completion=1),
            ]

            router = ModelRouter(config=primary, fallbacks=[fb])
            router.send_message([{"role": "user", "content": "hi"}])

        assert router.get_stats("primary")["primary"].error_count == 1

    def test_get_all_stats(self, primary, fallback):
        router = ModelRouter(config=primary, fallbacks=[fallback])
        all_stats = router.get_stats()
        assert "primary" in all_stats
        assert "fallback" in all_stats
        assert all_stats["primary"].avg_latency_ms == 0.0
