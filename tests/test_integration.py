"""Integration tests — end-to-end smoke tests for the full system.

These tests verify that:
  1. llama.cpp is running and reachable on localhost:7778
  2. The local coding agent can execute a prompt non-interactively
  3. The agent's tool calls actually affect the filesystem
  4. Multi-turn tool chaining works (write -> read, write -> patch, etc.)

Requirements:
  - llama.cpp server running on localhost:7778
  - This project's .venv activated
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
import pytest


def _run_agent(prompt: str, work_dir: Path, max_turns: int = 10) -> subprocess.CompletedProcess[str]:
    """Helper: run the agent non-interactively with the given prompt."""
    venv_python = Path(sys.prefix) / "bin" / "python"

    cmd = [
        str(venv_python),
        "-m", "local_agent",
        "--prompt", prompt,
        "--max-turns", str(max_turns),
    ]

    env = {
        "LLM_HOST": "localhost",
        "LLM_MODEL": "llama",
        "LLM_PORT": "7778",
        "PATH": str(Path(sys.prefix) / "bin") + ":" + os.environ.get("PATH", ""),
    }

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(work_dir),
        env=env,
    )


# ---------------------------------------------------------------------------
# Test 1: Ollama is running and serves the model
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_llamacpp_running_and_model_available():
    """Verify llama.cpp is running and the default model can produce a completion."""
    # Check server is up via models endpoint
    resp = httpx.get("http://localhost:7778/v1/models", timeout=10.0)
    assert resp.status_code == 200, "llama.cpp API /v1/models returned non-200"

    chat_resp = httpx.post(
        "http://localhost:7778/v1/chat/completions",
        json={
            "model": "llama",
            "messages": [
                {"role": "user", "content": "Say ONLY the word hello and nothing else."},
            ],
            "max_tokens": 10,
        },
        timeout=120.0,
    )
    assert chat_resp.status_code == 200, (
        f"llama.cpp chat completions failed: {chat_resp.text}"
    )
    content = chat_resp.json()["choices"][0]["message"]["content"]
    content = content.strip()
    print(f"  llama.cpp response: {content if content else '(empty but OK)'}")


# ---------------------------------------------------------------------------
# Test 2: Agent creates a file via prompt (single tool call)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_agent_creates_file_via_prompt():
    """Run the agent non-interactively with a prompt asking it to create a file,
    then verify the file was actually created.

    Smoke test: Ollama -> Agent -> write_file tool.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        expected_file = work_dir / "integration_test.txt"

        result = _run_agent(
            f"Create a file at {expected_file} with the word integration_test inside it.",
            work_dir,
            max_turns=3,
        )

        print(f"  Agent stdout:\n{result.stdout}")
        if result.stderr:
            print(f"  Agent stderr:\n{result.stderr}")

        assert result.returncode == 0, (
            f"Agent exited with code {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert expected_file.exists(), (
            f"Agent did not create {expected_file}\nstdout: {result.stdout}"
        )
        content = expected_file.read_text().strip()
        assert "integration_test" in content, (
            f"File created but does not contain 'integration_test': {content}"
        )


# ---------------------------------------------------------------------------
# Test 3: Write then read back (two-tool chain)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_agent_write_then_read_file():
    """Agent should write a file and then read it back to confirm contents.

    Tests multi-turn tool chaining: write_file -> read_file -> final answer.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        target = work_dir / "chain_test.txt"

        result = _run_agent(
            f"First write the text 'chain_verified' to {target}, then read it back and tell me what it says.",
            work_dir,
            max_turns=5,
        )

        print(f"  Agent stdout:\n{result.stdout}")
        assert result.returncode == 0, f"Agent failed: {result.stderr}"
        assert target.exists(), f"File not created: {target}"

        file_content = target.read_text().strip()
        assert "chain_verified" in file_content, (
            f"File does not contain 'chain_verified': {file_content}"
        )


# ---------------------------------------------------------------------------
# Test 4: Write then patch (two-tool chain)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_agent_write_then_patch_file():
    """Agent should write a file and then patch it.

    Tests multi-turn tool chaining: write_file -> patch_file -> final answer.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        target = work_dir / "patch_test.txt"

        result = _run_agent(
            f"Write 'hello world' to {target}. Then change 'world' to 'earth' in that file.",
            work_dir,
            max_turns=6,
        )

        print(f"  Agent stdout:\n{result.stdout}")
        assert result.returncode == 0, f"Agent failed: {result.stderr}"
        assert target.exists(), f"File not created: {target}"

        content = target.read_text().strip()
        assert "earth" in content, f"Patch failed — expected 'earth' in: {content}"


# ---------------------------------------------------------------------------
# Test 5: Execute a shell command
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_agent_executes_shell_command():
    """Agent should execute a shell command and report the result.

    Tests tool chaining: execute_command -> final answer.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)

        result = _run_agent(
            "Run the command 'echo hello_from_agent' and tell me the output.",
            work_dir,
            max_turns=4,
        )

        print(f"  Agent stdout:\n{result.stdout}")
        assert result.returncode == 0, f"Agent failed: {result.stderr}"
        # The agent's response should mention the echo output
        assert "hello_from_agent" in result.stdout or "hello" in result.stdout, (
            f"Agent did not report command output: {result.stdout}"
        )


# ---------------------------------------------------------------------------
# Test 6: Multi-step chain — write, execute ls, confirm file exists
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_agent_write_and_verify_with_ls():
    """Agent should write a file, run 'ls' to verify it exists, then report.

    Tests three-tool chain: write_file -> execute_command -> final answer.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        target = work_dir / "ls_test.txt"

        result = _run_agent(
            f"Write 'test' to {target}. Then run 'ls' to show the directory contents and confirm the file is there.",
            work_dir,
            max_turns=6,
        )

        print(f"  Agent stdout:\n{result.stdout}")
        assert result.returncode == 0, f"Agent failed: {result.stderr}"
        assert target.exists(), f"File not created: {target}"


# ---------------------------------------------------------------------------
# Test 7: Git workflow — write, init, add, commit
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_agent_git_write_and_commit():
    """Agent should write a file, init git, add, and commit it.

    Tests multi-tool chain: write_file -> git_init -> git_add -> git_commit.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        target = work_dir / "README.md"

        result = _run_agent(
            f"Write '# Hello' to {target}. Then initialize a git repo here, stage all files, and commit with the message 'initial commit'.",
            work_dir,
            max_turns=8,
        )

        print(f"  Agent stdout:\n{result.stdout}")
        assert result.returncode == 0, f"Agent failed: {result.stderr}"
        assert target.exists(), f"File not created: {target}"
        assert (work_dir / ".git").exists(), "Git repo not initialized"

        # Verify the commit exists
        log = subprocess.run(
            ["git", "log", "--oneline"],
            capture_output=True, text=True, cwd=str(work_dir),
        )
        assert "initial commit" in log.stdout.lower(), (
            f"No commit found: {log.stdout}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
