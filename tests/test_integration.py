"""Integration tests — end-to-end smoke tests for the full system.

These tests verify that:
  1. Ollama is running and reachable
  2. The local coding agent can execute a prompt non-interactively
  3. The agent's tool calls actually affect the filesystem

Requirements:
  - Ollama running on localhost:11434 with qwen3.5:4b pulled
  - This project's .venv activated
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
import pytest


# ---------------------------------------------------------------------------
# Test 1: Ollama is running and serves the model
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_ollama_running_and_model_available():
    """Verify Ollama is running and qwen3.5:4b can produce a completion."""
    # Check the server is reachable
    resp = httpx.get("http://localhost:11434/api/tags", timeout=10.0)
    assert resp.status_code == 200, "Ollama API /api/tags returned non-200"

    # Check the model is listed
    data = resp.json()
    model_names = [m["name"] for m in data.get("models", [])]
    assert any(
        "qwen3.5:4b" in name for name in model_names
    ), f"qwen3.5:4b not found in models: {model_names}"

    # Do a live "ollama run" via the chat API
    chat_resp = httpx.post(
        "http://localhost:11434/v1/chat/completions",
        json={
            "model": "qwen3.5:4b",
            "messages": [
                {"role": "user", "content": "Say ONLY the word hello and nothing else."},
            ],
            "max_tokens": 10,
        },
        timeout=120.0,
    )
    assert chat_resp.status_code == 200, (
        f"Ollama chat completions failed: {chat_resp.text}"
    )
    content = chat_resp.json()["choices"][0]["message"]["content"]
    assert len(content) > 0, "Ollama returned empty content"
    print(f"  Ollama response: {content.strip()}")


# ---------------------------------------------------------------------------
# Test 2: Agent executes a prompt and creates a file
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_agent_creates_file_via_prompt():
    """Run the agent non-interactively with a prompt asking it to create a file,
    then verify the file was actually created.

    This is the full end-to-end smoke test: Ollama -> Agent -> write_file tool.
    """
    # Create a temporary working directory for the agent to use
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        expected_file = work_dir / "integration_test.txt"

        # Build the command
        venv_python = Path(sys.prefix) / "bin" / "python"
        src_dir = Path(__file__).resolve().parent.parent / "src"

        cmd = [
            str(venv_python),
            "-m", "local_agent",
            "--prompt",
            f"Create a file at {expected_file} with the word integration_test inside it.",
            "--max-turns", "3",
        ]

        env = {
            "LLM_HOST": "localhost",
            "LLM_MODEL": "qwen3.5:4b",
            "LLM_PORT": "11434",
            "PATH": str(Path(sys.prefix) / "bin") + ":" + os.environ.get("PATH", ""),
        }

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(work_dir),
            env=env,
        )

        # Print output for debugging
        print(f"  Agent stdout:\n{result.stdout}")
        if result.stderr:
            print(f"  Agent stderr:\n{result.stderr}")

        # The agent should have exited cleanly
        assert result.returncode == 0, (
            f"Agent exited with code {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

        # Verify the file was created
        assert expected_file.exists(), (
            f"Agent did not create {expected_file}\n"
            f"stdout: {result.stdout}"
        )

        # Verify the file contains the expected word
        content = expected_file.read_text().strip()
        assert "integration_test" in content, (
            f"File created but does not contain 'integration_test': {content}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
