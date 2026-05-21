"""Tests for ModelRouter — LLM client with streaming."""

from unittest.mock import MagicMock, patch

import pytest
import httpx

from local_agent.config import LLMConfig
from local_agent.model_router import ModelRouter


class TestModelRouterSend:
    """Test send_message returns a string completion."""

    def _make_router(self, deterministic=False):
        config = LLMConfig(model="test-model", deterministic=deterministic)
        return ModelRouter(config)

    def test_send_message_returns_string(self):
        router = self._make_router()
        with patch.object(httpx, "post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "Hello world"}}]
            }
            mock_post.return_value = mock_resp

            result = router.send_message([{"role": "user", "content": "hi"}])
            assert result == "Hello world"

    def test_send_message_sends_correct_endpoint(self):
        router = self._make_router()
        with patch.object(httpx, "post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "ok"}}]
            }
            mock_post.return_value = mock_resp

            router.send_message([{"role": "user", "content": "hi"}])
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "http://localhost:11434/v1/chat/completions"

    def test_send_message_sends_model(self):
        router = self._make_router()
        with patch.object(httpx, "post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "ok"}}]
            }
            mock_post.return_value = mock_resp

            router.send_message([{"role": "user", "content": "hi"}])
            json_data = mock_post.call_args[1]["json"]
            assert json_data["model"] == "test-model"

    def test_send_message_sends_messages(self):
        router = self._make_router()
        msgs = [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hello"},
        ]
        with patch.object(httpx, "post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "ok"}}]
            }
            mock_post.return_value = mock_resp

            router.send_message(msgs)
            json_data = mock_post.call_args[1]["json"]
            assert json_data["messages"] == msgs

    def test_deterministic_sets_temperature_and_seed(self):
        router = self._make_router(deterministic=True)
        with patch.object(httpx, "post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "ok"}}]
            }
            mock_post.return_value = mock_resp

            router.send_message([{"role": "user", "content": "hi"}])
            json_data = mock_post.call_args[1]["json"]
            assert json_data["temperature"] == 0
            assert json_data["seed"] == 0

    def test_non_deterministic_no_temperature_or_seed(self):
        router = self._make_router(deterministic=False)
        with patch.object(httpx, "post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "ok"}}]
            }
            mock_post.return_value = mock_resp

            router.send_message([{"role": "user", "content": "hi"}])
            json_data = mock_post.call_args[1]["json"]
            assert "temperature" not in json_data
            assert "seed" not in json_data

    def test_connection_error_raises(self):
        router = self._make_router()
        with patch.object(httpx, "post", side_effect=httpx.ConnectError("no host")):
            with pytest.raises(httpx.ConnectError):
                router.send_message([{"role": "user", "content": "hi"}])


class TestModelRouterStream:
    """Test stream_message yields content chunks."""

    def test_stream_yields_chunks(self):
        config = LLMConfig(model="test-model")
        router = ModelRouter(config)

        # Simulate an SSE stream response
        chunks = [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
            b'data: {"choices": [{"delta": {"content": " world"}}]}\n\n',
            b'data: [DONE]\n\n',
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(chunks)
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch.object(httpx.Client, "stream", return_value=mock_response) as mock_stream:
            results = list(router.stream_message([{"role": "user", "content": "hi"}]))
            assert results == ["Hello", " world"]

    def test_stream_ignores_empty_content(self):
        config = LLMConfig(model="test-model")
        router = ModelRouter(config)

        chunks = [
            b'data: {"choices": [{"delta": {"content": ""}}]}\n\n',
            b'data: {"choices": [{"delta": {"content": "real"}}]}\n\n',
            b'data: [DONE]\n\n',
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(chunks)
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch.object(httpx.Client, "stream", return_value=mock_response):
            results = list(router.stream_message([{"role": "user", "content": "hi"}]))
            assert results == ["real"]
