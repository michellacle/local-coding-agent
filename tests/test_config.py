"""Tests for config — ConfigManager with YAML, env interpolation, hot-reload."""

import os
import time

import pytest

from local_agent.config import ConfigManager, ConfigValidationError


class TestConfigManagerDefaults:
    """Test default config handling."""

    def test_empty_config(self):
        mgr = ConfigManager()
        assert mgr.to_dict() == {}

    def test_default_config(self):
        defaults = {"llm": {"provider": "ollama-local", "model": "qwen3.5:4b"}}
        mgr = ConfigManager(default_config=defaults)
        assert mgr.get("llm.provider") == "ollama-local"
        assert mgr.get("llm.model") == "qwen3.5:4b"

    def test_get_missing_key(self):
        mgr = ConfigManager()
        assert mgr.get("nonexistent") is None
        assert mgr.get("nonexistent", "fallback") == "fallback"

    def test_get_nested_key(self):
        defaults = {"a": {"b": {"c": 42}}}
        mgr = ConfigManager(default_config=defaults)
        assert mgr.get("a.b.c") == 42
        assert mgr.get("a.b") == {"c": 42}

    def test_set_value(self):
        mgr = ConfigManager()
        mgr.set("llm.provider", "openai")
        assert mgr.get("llm.provider") == "openai"

    def test_set_nested(self):
        mgr = ConfigManager()
        mgr.set("a.b.c", 100)
        assert mgr.get("a.b.c") == 100

    def test_get_section(self):
        defaults = {"llm": {"provider": "ollama", "model": "qwen"}, "db": {"path": "/tmp"}}
        mgr = ConfigManager(default_config=defaults)
        section = mgr.get_section("llm")
        assert section == {"provider": "ollama", "model": "qwen"}

    def test_get_section_missing(self):
        mgr = ConfigManager()
        assert mgr.get_section("missing") == {}


class TestConfigManagerDict:
    """Test loading from dicts."""

    def test_load_dict(self):
        mgr = ConfigManager()
        mgr.load_dict({"key": "value"}, priority=1)
        assert mgr.get("key") == "value"

    def test_load_dict_override_defaults(self):
        defaults = {"llm": {"provider": "ollama", "model": "qwen"}}
        mgr = ConfigManager(default_config=defaults)
        mgr.load_dict({"llm": {"model": "mistral"}}, priority=1)
        assert mgr.get("llm.provider") == "ollama"  # from defaults
        assert mgr.get("llm.model") == "mistral"  # overridden

    def test_load_dict_priority(self):
        mgr = ConfigManager()
        mgr.load_dict({"key": "low"}, priority=1)
        mgr.load_dict({"key": "high"}, priority=2)
        assert mgr.get("key") == "high"

    def test_deep_merge(self):
        merged = ConfigManager._deep_merge(
            {"a": {"b": 1, "c": 2}},
            {"a": {"b": 10, "d": 3}},
        )
        assert merged == {"a": {"b": 10, "c": 2, "d": 3}}


class TestConfigManagerYAML:
    """Test YAML file loading."""

    def test_load_yaml_file(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("llm:\n  provider: ollama-local\n  model: qwen3.5:4b\n")

        mgr = ConfigManager()
        mgr.load_file(str(config_file))

        assert mgr.get("llm.provider") == "ollama-local"
        assert mgr.get("llm.model") == "qwen3.5:4b"

    def test_load_missing_file(self):
        mgr = ConfigManager()
        with pytest.raises(FileNotFoundError):
            mgr.load_file("/nonexistent/config.yaml")

    def test_load_invalid_yaml(self, tmp_path):
        config_file = tmp_path / "bad.yaml"
        config_file.write_text(": : : invalid yaml [[[")

        mgr = ConfigManager()
        with pytest.raises(ConfigValidationError):
            mgr.load_file(str(config_file))

    def test_load_non_mapping_yaml(self, tmp_path):
        config_file = tmp_path / "list.yaml"
        config_file.write_text("- item1\n- item2\n")

        mgr = ConfigManager()
        with pytest.raises(ConfigValidationError, match="must contain a YAML mapping"):
            mgr.load_file(str(config_file))

    def test_load_multiple_files_priority(self, tmp_path):
        base = tmp_path / "base.yaml"
        base.write_text("llm:\n  provider: ollama\n  model: qwen\n")

        override = tmp_path / "override.yaml"
        override.write_text("llm:\n  model: mistral\n")

        mgr = ConfigManager()
        mgr.load_file(str(base), priority=1)
        mgr.load_file(str(override), priority=2)

        assert mgr.get("llm.provider") == "ollama"  # from base
        assert mgr.get("llm.model") == "mistral"  # overridden


class TestConfigManagerEnvInterpolation:
    """Test environment variable interpolation."""

    def test_env_var_interpolation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "custom-ollama")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("llm:\n  provider: ${LLM_PROVIDER}\n")

        mgr = ConfigManager()
        mgr.load_file(str(config_file))
        assert mgr.get("llm.provider") == "custom-ollama"

    def test_env_var_with_default(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("llm:\n  model: ${MISSING_VAR:-default-model}\n")

        mgr = ConfigManager()
        mgr.load_file(str(config_file))
        assert mgr.get("llm.model") == "default-model"

    def test_env_var_missing_no_default(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("llm:\n  model: ${UNDEFINED_VAR}\n")

        mgr = ConfigManager()
        mgr.load_file(str(config_file))
        # Unset vars keep their literal form
        assert mgr.get("llm.model") == "${UNDEFINED_VAR}"

    def test_env_var_in_list(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_VAR", "value1")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("items:\n  - ${MY_VAR}\n  - static\n")

        mgr = ConfigManager()
        mgr.load_file(str(config_file))
        assert mgr.get("items") == ["value1", "static"]

    def test_no_interpolation(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: ${NOT_EXPANDED}\n")

        mgr = ConfigManager()
        mgr.load_file(str(config_file), interpolate_env=False)
        assert mgr.get("key") == "${NOT_EXPANDED}"


class TestConfigManagerReload:
    """Test hot-reload functionality."""

    def test_check_for_updates(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: initial\n")

        mgr = ConfigManager()
        mgr._reload_interval = 0  # Force immediate check
        mgr.load_file(str(config_file))

        # Modify the file
        time.sleep(0.01)
        config_file.write_text("key: updated\n")

        updated = mgr.check_for_updates()
        assert updated is True
        assert mgr.get("key") == "updated"

    def test_reload_callback(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: v1\n")

        mgr = ConfigManager()
        mgr._reload_interval = 0
        mgr.load_file(str(config_file))

        called = []
        mgr.on_change(lambda config: called.append(config.get("key")))

        time.sleep(0.01)
        config_file.write_text("key: v2\n")
        mgr.check_for_updates()

        assert "v2" in called

    def test_force_reload(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: v1\n")

        mgr = ConfigManager()
        mgr.load_file(str(config_file))

        time.sleep(0.01)
        config_file.write_text("key: v2\n")
        mgr.reload()
        assert mgr.get("key") == "v2"

    def test_reload_interval(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: v1\n")

        mgr = ConfigManager()
        mgr._reload_interval = 300
        mgr.load_file(str(config_file))

        # Should NOT check because interval hasn't passed
        updated = mgr.check_for_updates()
        assert updated is False


class TestConfigManagerExport:
    """Test config export."""

    def test_export_yaml(self, tmp_path):
        mgr = ConfigManager(default_config={"llm": {"provider": "ollama"}})
        export_path = str(tmp_path / "export.yaml")
        mgr.export_yaml(export_path)

        content = Path(export_path).read_text()
        assert "llm:" in content
        assert "provider: ollama" in content

    def test_to_dict(self):
        mgr = ConfigManager(default_config={"a": 1, "b": {"c": 2}})
        d = mgr.to_dict()
        assert d == {"a": 1, "b": {"c": 2}}


class TestPath:
    """Helper for Path import."""
    pass


# Need Path for tests
from pathlib import Path
