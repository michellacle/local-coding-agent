# Local Coding Agent

## Project Goal

Build a local-only coding agent grounded in proven agentic architectural patterns. The agent lives in the terminal, integrates deeply with git, supports multi-agent orchestration, and operates fully offline. It aims to rival Claude Code and OpenCode in capability while giving users full control over models, tools, and data.

## Architectural Requirement: Local-Only Models

This agent is designed exclusively for models that run locally on the user's hardware. No cloud LLM APIs. No external inference services. The entire stack must work offline.

**Hardware assumption:** Users have access to up to 48GB of GPU VRAM (e.g., dual RTX 3090s, single A6000, or similar). This enables running models like Qwen 3.6 27B, Llama 3.3 70B (quantized), or Mistral Large (quantized) entirely on-device.

**Implications:**
- Embedding models must also run locally (e.g., nomic-embed-text via Ollama)
- No fallback to cloud providers (OpenAI, Anthropic, etc.)
- Model selection assumes FP16/INT8 quantized models fitting within 48GB VRAM
- Context window planning targets 32K-128K tokens within GPU memory constraints
- All dependencies (vector store, model serving, inference) are self-contained

## Session Summary (May 21, 2026)

1. Extracted a 574-page PDF textbook ("Agentic Architectural Patterns for Building Multi-Agent Systems") to markdown using pymupdf.
2. Researched two how-to guides on thinksmart.life covering product requirements for coding agents and building agents from agentic patterns.
3. Built a comprehensive Product Requirements Document (19 functional requirements, 3-layer architecture, 8-week phased build plan) based on competitive analysis and the book's patterns.

## Current Artifacts

- `9781806029570.pdf` — Full textbook (574 pages)
- `9781806029570.md` — Extracted text from the textbook (~26K lines)
- `PRODUCT_REQUIREMENTS.md` — Complete PRD with requirements, architecture, and build plan

## Source Material

- **Build Product Requirements for a Coding Agent** — https://thinksmart.life/howto/build-product-requirements-coding-agent/
- **Build an AI Agent from Agentic Architectural Patterns** — https://thinksmart.life/howto/build-ai-agent-from-agentic-patterns/
- **Book:** *Agentic Architectural Patterns for Building Multi-Agent Systems* by Dr. Ali Arsanjani & Juan Pablo Bustos (Packt, 2026)

## Next Steps

- Review and refine PRODUCT_REQUIREMENTS.md
- Begin Phase 1: Model router, terminal UI, file operations, and LLM client
