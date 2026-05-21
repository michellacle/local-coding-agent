"""Configuration dataclasses for the local coding agent."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    """LLM connection configuration.

    Attributes:
        host: Hostname of the LLM server.
        model: Model name to use.
        port: Port of the LLM server.
        deterministic: If True, forces temperature=0 and seed=0.
    """

    host: str = "localhost"
    model: str = "qwen3.5:0.8b"
    port: int = 11434
    deterministic: bool = False

    @property
    def base_url(self) -> str:
        """Build the OpenAI-compatible base URL."""
        return f"http://{self.host}:{self.port}/v1"

    @classmethod
    def ollama(cls, host: str, model: str) -> "LLMConfig":
        """Factory: create config for a specific Ollama host and model."""
        return cls(host=host, model=model)

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Factory: create config from environment variables.

        Reads LLM_HOST, LLM_MODEL, LLM_PORT, LLM_DETERMINISTIC.
        Falls back to defaults if not set.
        """
        return cls(
            host=os.environ.get("LLM_HOST", "localhost"),
            model=os.environ.get("LLM_MODEL", "qwen3.5:0.8b"),
            port=int(os.environ.get("LLM_PORT", "11434")),
            deterministic=os.environ.get("LLM_DETERMINISTIC", "").lower() == "true",
        )


@dataclass
class AppConfig:
    """Top-level application configuration.

    Attributes:
        llm: LLM connection settings.
        work_dir: Working directory for file/terminal operations.
        toolsets: Which tool sets to register ("file", "terminal", "git").
    """

    llm: LLMConfig
    work_dir: Path
    toolsets: list[str] = field(default_factory=lambda: ["file", "terminal", "git"])
