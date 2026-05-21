"""Tests for configuration dataclasses and factories."""

import os
from pathlib import Path

import pytest

from local_agent.config import LLMConfig, AppConfig


class TestLLMConfigDefaults:
    """Test default values for LLMConfig."""

    def test_default_host(self):
        config = LLMConfig()
        assert config.host == "localhost"

    def test_default_model(self):
        config = LLMConfig()
        assert config.model == "qwen3.5:0.8b"

    def test_default_port(self):
        config = LLMConfig()
        assert config.port == 11434

    def test_default_deterministic(self):
        config = LLMConfig()
        assert config.deterministic is False

    def test_base_url_property_default(self):
        config = LLMConfig()
        assert config.base_url == "http://localhost:11434/v1"

    def test_base_url_property_custom(self):
        config = LLMConfig(host="192.168.1.100", port=9090)
        assert config.base_url == "http://192.168.1.100:9090/v1"

    def test_custom_values(self):
        config = LLMConfig(host="myhost", model="mistral:7b", port=8080, deterministic=True)
        assert config.host == "myhost"
        assert config.model == "mistral:7b"
        assert config.port == 8080
        assert config.deterministic is True


class TestLLMConfigFactory:
    """Test factory methods for LLMConfig."""

    def test_ollama_factory(self):
        config = LLMConfig.ollama(host="127.0.0.1", model="llama3.2:3b")
        assert config.host == "127.0.0.1"
        assert config.model == "llama3.2:3b"
        assert config.port == 11434

    def test_from_env_defaults(self):
        config = LLMConfig.from_env()
        assert config.host == "localhost"
        assert config.model == "qwen3.5:0.8b"
        assert config.port == 11434

    def test_from_env_custom(self, monkeypatch):
        monkeypatch.setenv("LLM_HOST", "remote-host")
        monkeypatch.setenv("LLM_MODEL", "custom-model")
        monkeypatch.setenv("LLM_PORT", "9999")
        monkeypatch.setenv("LLM_DETERMINISTIC", "true")
        config = LLMConfig.from_env()
        assert config.host == "remote-host"
        assert config.model == "custom-model"
        assert config.port == 9999
        assert config.deterministic is True


class TestAppConfig:
    """Test AppConfig dataclass."""

    def test_basic_creation(self):
        llm = LLMConfig()
        app = AppConfig(llm=llm, work_dir=Path("/tmp/test"), toolsets=["file", "terminal"])
        assert app.llm is llm
        assert app.work_dir == Path("/tmp/test")
        assert app.toolsets == ["file", "terminal"]

    def test_default_toolsets(self):
        llm = LLMConfig()
        app = AppConfig(llm=llm, work_dir=Path("/tmp/test"))
        assert app.toolsets == ["file", "terminal", "git"]
