"""Tests for /provider slash command — view and change LLM provider config."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from local_agent.terminal_ui import TerminalUI
from local_agent.config import AppConfig, LLMConfig


@pytest.fixture
def ui(tmp_path):
    """Create a TerminalUI with a mock agent and temp work dir."""
    agent = MagicMock()
    config = AppConfig(work_dir=str(tmp_path))
    return TerminalUI(agent, config)


@pytest.fixture
def custom_ui(tmp_path):
    """TerminalUI with a custom provider config."""
    agent = MagicMock()
    llm = LLMConfig(
        model="qwen3.5:32b",
        host="moneymaker",
        port=8090,
        deterministic=True,
    )
    config = AppConfig(work_dir=str(tmp_path), llm=llm)
    return TerminalUI(agent, config)


class TestProviderShow:
    """Test /provider with no args shows current config."""

    def test_shows_current_config(self, ui):
        result = ui._cmd_provider("")
        assert "LLM Provider" in result
        assert "Model" in result
        assert "Host" in result
        assert "Port" in result
        assert "Base URL" in result

    def test_shows_default_values(self, ui):
        result = ui._cmd_provider("")
        assert "llama" in result
        assert "localhost" in result
        assert "7778" in result

    def test_shows_custom_values(self, custom_ui):
        result = custom_ui._cmd_provider("")
        assert "qwen3.5:32b" in result
        assert "moneymaker" in result
        assert "8090" in result

    def test_shows_deterministic(self, custom_ui):
        result = custom_ui._cmd_provider("")
        assert "True" in result

    def test_shows_derived_base_url(self, custom_ui):
        result = custom_ui._cmd_provider("")
        assert "http://moneymaker:8090/v1" in result


class TestProviderSetHost:
    """Test /provider host <value>."""

    def test_set_host(self, ui):
        result = ui._cmd_provider("host minadioro")
        assert ui.config.llm.host == "minadioro"
        assert "minadioro" in result
        assert "✓" in result or "green" in result

    def test_set_host_with_http(self, ui):
        result = ui._cmd_provider("host http://papia.tailde85bf.ts.net:8880")
        assert ui.config.llm.host == "http://papia.tailde85bf.ts.net:8880"

    def test_set_host_no_value(self, ui):
        result = ui._cmd_provider("host")
        assert "Error" in result or "Usage" in result

    def test_host_updates_base_url(self, ui):
        ui._cmd_provider("host myserver")
        assert "myserver" in ui.config.llm.base_url


class TestProviderSetPort:
    """Test /provider port <N>."""

    def test_set_port(self, ui):
        result = ui._cmd_provider("port 8090")
        assert ui.config.llm.port == 8090
        assert "8090" in result

    def test_set_port_validates(self, ui):
        result = ui._cmd_provider("port abc")
        assert "Error" in result

    def test_set_port_no_value(self, ui):
        result = ui._cmd_provider("port")
        assert "Error" in result or "Usage" in result

    def test_port_updates_base_url(self, ui):
        ui._cmd_provider("port 11434")
        assert "11434" in ui.config.llm.base_url


class TestProviderSetModel:
    """Test /provider model <name>."""

    def test_set_model(self, ui):
        result = ui._cmd_provider("model qwen3.5:9b")
        assert ui.config.llm.model == "qwen3.5:9b"
        assert "qwen3.5:9b" in result

    def test_set_model_no_value(self, ui):
        result = ui._cmd_provider("model")
        assert "Error" in result or "Usage" in result


class TestProviderSetUrl:
    """Test /provider url <base>."""

    def test_set_url(self, ui):
        result = ui._cmd_provider("url http://minadioro:11434")
        assert ui.config.llm.host == "http://minadioro:11434"
        assert ui.config.llm.port == 0

    def test_set_url_no_value(self, ui):
        result = ui._cmd_provider("url")
        assert "Error" in result or "Usage" in result

    def test_url_overrides_host(self, ui):
        ui.config.llm.host = "localhost"
        ui.config.llm.port = 7778
        ui._cmd_provider("url http://remote-server:9000/v1")
        assert ui.config.llm.host == "http://remote-server:9000/v1"
        assert ui.config.llm.port == 0


class TestProviderUnknownAction:
    """Test /provider with unknown subcommand."""

    def test_unknown_action(self, ui):
        result = ui._cmd_provider("foo bar")
        assert "Unknown" in result
        assert "foo" in result.lower()

    def test_unknown_action_shows_usage(self, ui):
        result = ui._cmd_provider("invalid")
        assert "host" in result or "port" in result or "model" in result


class TestProviderDispatch:
    """Test /provider is wired into the dispatcher."""

    def test_dispatch_show(self, ui):
        result = ui.handle_special_command("/provider")
        assert "LLM Provider" in result or "Provider" in result

    def test_dispatch_set(self, ui):
        ui.handle_special_command("/provider port 12345")
        assert ui.config.llm.port == 12345
