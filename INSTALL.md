# Local Coding Agent — Installation & Usage Guide

## Quick Start

1. Clone the repo
2. Create a virtual environment
3. Install dependencies
4. Start the agent

That's it. Full instructions below.

## Prerequisites

- **Python 3.12+**
- **Git** (for git operations)
- **Ollama** (local LLM serving) — download from https://ollama.ai
- **GPU with up to 48GB VRAM** (e.g., dual RTX 3090s) — for running large local models

## Installation

### 1. Clone the Repository

```bash
git clone <repo-url>
cd local-coding-agent
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -e .
```

This installs:
- `httpx` — HTTP client for LLM API calls
- `rich` — terminal UI with streaming output
- `pydantic` — configuration validation
- `pytest` + `pytest-mock` — testing
- `mypy` — strict type checking

### 4. Install Ollama and Pull a Model

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (adjust based on your GPU)
ollama pull qwen3.5:0.8b   # Small, fast, works on most GPUs
ollama pull qwen3.5:8b     # Medium quality
ollama pull qwen3.6:27b    # High quality, needs ~48GB VRAM
```

### 5. Run the Tests

Before using the agent, verify everything works:

```bash
pytest tests/ -v
```

All 86 tests should pass.

### 6. Run the Agent

```bash
  python -m local_agent
```

The agent will start in your terminal with a prompt:

```
Local Coding Agent
Working directory: /your/current/dir
Type your request or 'quit' to exit.

You >>>
```

## Configuration

### Environment Variables

Set these before running the agent (or add to your shell profile):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_HOST` | `localhost` | LLM server hostname |
| `LLM_MODEL` | `qwen3.5:0.8b` | Model name |
| `LLM_PORT` | `11434` | Ollama port |
| `LLM_DETERMINISTIC` | `false` | Force temperature=0 for reproducible output |

Example:

```bash
export LLM_HOST=localhost
export LLM_MODEL=qwen3.6:27b
python -m local_agent
```

### Programmatic Configuration

```python
from local_agent.config import LLMConfig, AppConfig
from local_agent.model_router import ModelRouter
from local_agent.tool_registry import ToolRegistry
from local_agent.agent_core import AgentCore

config = AppConfig(
    llm=LLMConfig(host="localhost", model="qwen3.6:27b"),
    work_dir="/your/project",
)

router = ModelRouter(config.llm)
registry = ToolRegistry()
agent = AgentCore(router, registry)
```

## Usage Examples

### Ask a Question

```
You >>> What files are in the current directory?
```

The agent will use its tools to read your codebase and respond.

### Run a Command

```
You >>> Run pytest on the tests directory
```

The agent will execute `pytest tests/` and return the output.

### Make Code Changes

```
You >>> Fix the bug where read_file returns empty for missing files
```

The agent will read the code, identify the issue, and make the edit.

### Multi-Step Tasks

```
You >>> Create a new module for database operations with connection pooling
```

The agent will plan, create files, write code, and run tests.

## Project Structure

```
src/local_agent/
  config.py          # Configuration dataclasses
  model_router.py    # LLM client with streaming
  tool_registry.py   # Tool schemas and execution
  agent_core.py      # Main agent loop
  terminal_ui.py     # Rich-based CLI
  tools/
    file_tools.py    # File read/write/patch
    terminal_tools.py # Shell command execution
    git_tools.py     # Git operations
tests/               # Unit tests for all modules
```

## Quality Assurance

All code is built with:
- **Strict TDD** — every module was built test-first (RED-GREEN-REFACTOR)
- **Full type hints** — mypy --strict passes with zero issues
- **Comprehensive tests** — 86 unit tests, zero real network calls
- **Clean separation of concerns** — dependency injection throughout

Run the full quality check:

```bash
pytest tests/ -v     # All tests
mypy src/ --strict   # Type checking
```

## Troubleshooting

### Model Not Found

```bash
ollama list          # Check installed models
ollama pull qwen3.5:0.8b  # Install a model
```

### Connection Refused

Make sure Ollama is running:

```bash
ollama serve &       # Start Ollama
curl http://localhost:11434/api/tags  # Verify it's up
```

### Out of Memory on Large Models

If your GPU can't fit the model, try a smaller one:

```bash
ollama pull qwen3.5:0.8b   # ~600MB
ollama pull qwen3.5:1.8b   # ~1.2GB
ollama pull qwen3.5:4b     # ~2.5GB
```

## Next Steps

- Add more tools (browser automation, RAG, multi-agent orchestration)
- Configure additional models for the router
- Create custom skills for your project
- Set up CI/CD with the test suite
