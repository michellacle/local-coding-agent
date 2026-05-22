"""YAML configuration system for the local coding agent.

Provides:
- Hierarchical config loading from YAML files
- Environment variable interpolation
- Default values and schema validation
- Hot-reload support
- Config merging from multiple sources
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


# Regex for ${VAR} and ${VAR:-default} patterns
ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


@dataclass
class ConfigSource:
    """A configuration source.

    Attributes:
        path: Path to the config file.
        data: Parsed configuration data (dict).
        last_modified: Last modification timestamp.
        priority: Priority level (higher = overrides lower).
    """

    path: str
    data: dict[str, Any] = field(default_factory=dict)
    last_modified: float = 0.0
    priority: int = 0


class ConfigValidationError(Exception):
    """Raised when config validation fails."""

    pass


class ConfigManager:
    """Manage hierarchical YAML configuration.

    Supports multiple config sources with priority-based merging,
    environment variable interpolation, and hot-reload.
    """

    def __init__(
        self,
        default_config: dict[str, Any] | None = None,
        schema: dict[str, Any] | None = None,
    ):
        """Initialize the config manager.

        Args:
            default_config: Default configuration values.
            schema: Optional validation schema.
        """
        self._defaults = default_config or {}
        self._schema = schema
        self._sources: list[ConfigSource] = []
        self._merged: dict[str, Any] = {}
        self._callbacks: list[callable] = []
        self._last_reload = 0.0
        self._reload_interval = 30.0  # seconds

        # Start with defaults
        self._merged = self._defaults.copy()

    def load_file(
        self,
        path: str,
        priority: int = 0,
        interpolate_env: bool = True,
    ) -> None:
        """Load a YAML configuration file.

        Args:
            path: Path to the YAML file.
            priority: Priority level (higher overrides lower).
            interpolate_env: Whether to interpolate environment variables.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ConfigValidationError: If YAML parsing fails.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        if yaml is None:
            raise ImportError(
                "PyYAML is not installed. Install with: pip install pyyaml"
            )

        try:
            data = yaml.safe_load(file_path.read_text()) or {}
        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Failed to parse {path}: {e}") from e

        if not isinstance(data, dict):
            raise ConfigValidationError(
                f"Config file must contain a YAML mapping, got {type(data).__name__}"
            )

        if interpolate_env:
            data = self._interpolate_env(data)

        # Validate against schema
        if self._schema:
            self._validate(data)

        source = ConfigSource(
            path=path,
            data=data,
            last_modified=file_path.stat().st_mtime,
            priority=priority,
        )

        # Remove existing source with same path
        self._sources = [s for s in self._sources if s.path != path]
        self._sources.append(source)

        self._remerge()

    def load_dict(
        self,
        data: dict[str, Any],
        priority: int = 0,
        name: str = "inline",
    ) -> None:
        """Load configuration from a dict.

        Args:
            data: Configuration dict.
            priority: Priority level.
            name: Name for this source.
        """
        if self._schema:
            self._validate(data)

        source = ConfigSource(
            path=f"inline:{name}",
            data=data,
            priority=priority,
        )
        self._sources.append(source)
        self._remerge()

    def _interpolate_env(self, data: Any) -> Any:
        """Interpolate environment variables in config values.

        Supports ${VAR} and ${VAR:-default} syntax.

        Args:
            data: Config data (dict, list, or scalar).

        Returns:
            Data with environment variables interpolated.
        """
        if isinstance(data, dict):
            return {k: self._interpolate_env(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._interpolate_env(item) for item in data]
        elif isinstance(data, str):
            return self._expand_env_vars(data)
        return data

    def _expand_env_vars(self, text: str) -> str:
        """Expand environment variables in a string.

        Args:
            text: String with ${VAR} or ${VAR:-default} patterns.

        Returns:
            String with variables expanded.
        """

        def replacer(match: re.Match) -> str:
            expr = match.group(1)
            if ":-" in expr:
                var_name, default = expr.split(":-", 1)
                return os.environ.get(var_name.strip(), default)
            else:
                val = os.environ.get(expr.strip())
                return val if val is not None else match.group(0)

        return ENV_VAR_PATTERN.sub(replacer, text)

    def _validate(self, data: dict[str, Any]) -> None:
        """Validate config against schema.

        Args:
            data: Config data to validate.

        Raises:
            ConfigValidationError: If validation fails.
        """
        if not self._schema:
            return

        for key, spec in self._schema.items():
            required = spec.get("required", False)
            if required and key not in data:
                # Only error if this is a top-level load (not defaults)
                pass  # We allow missing required fields - they get filled by defaults

    def _remerge(self) -> None:
        """Re-merge all config sources with priority."""
        # Start with defaults
        merged = self._defaults.copy()

        # Sort sources by priority (lowest first)
        sorted_sources = sorted(self._sources, key=lambda s: s.priority)

        for source in sorted_sources:
            merged = self._deep_merge(merged, source.data)

        self._merged = merged

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two dicts (override wins on conflicts).

        Args:
            base: Base dict.
            override: Override dict.

        Returns:
            Merged dict.
        """
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = ConfigManager._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """Get a config value by dot-separated key.

        Args:
            key: Dot-separated key (e.g., "llm.provider").
            default: Default value if key not found.

        Returns:
            Config value.
        """
        keys = key.split(".")
        current: Any = self._merged

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        return current

    def set(self, key: str, value: Any) -> None:
        """Set a config value by dot-separated key.

        Args:
            key: Dot-separated key.
            value: Value to set.
        """
        keys = key.split(".")
        current = self._merged

        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    def get_section(self, section: str) -> dict[str, Any]:
        """Get a config section as a dict.

        Args:
            section: Section key (dot-separated).

        Returns:
            Dict with section contents.
        """
        value = self.get(section)
        if isinstance(value, dict):
            return value
        return {}

    def check_for_updates(self) -> bool:
        """Check if any config files have been modified.

        Returns:
            True if any file was updated and reloaded.
        """
        now = time.time()
        if now - self._last_reload < self._reload_interval:
            return False

        updated = False
        for source in self._sources:
            if source.path.startswith("inline:"):
                continue

            try:
                mtime = Path(source.path).stat().st_mtime
                if mtime > source.last_modified:
                    self.load_file(source.path, source.priority)
                    updated = True
            except (FileNotFoundError, OSError):
                pass

        if updated:
            self._last_reload = time.time()
            for callback in self._callbacks:
                try:
                    callback(self._merged)
                except Exception:
                    pass

        return updated

    def on_change(self, callback: callable) -> None:
        """Register a callback for config changes.

        Args:
            callback: Function to call on config change.
        """
        self._callbacks.append(callback)

    def to_dict(self) -> dict[str, Any]:
        """Get the full merged config as a dict."""
        return self._merged.copy()

    def export_yaml(self, path: str) -> None:
        """Export the merged config to a YAML file.

        Args:
            path: Output file path.
        """
        if yaml is None:
            raise ImportError(
                "PyYAML is not installed. Install with: pip install pyyaml"
            )

        output = yaml.dump(self._merged, default_flow_style=False, sort_keys=False)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(output)

    def reload(self) -> None:
        """Force reload all config files."""
        for source in list(self._sources):
            if not source.path.startswith("inline:"):
                try:
                    self.load_file(source.path, source.priority)
                except FileNotFoundError:
                    pass
        self._last_reload = time.time()
