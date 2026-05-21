# Local Coding Agent

## Project Goal

Build a local-first, provider-agnostic coding agent grounded in proven agentic architectural patterns. The agent lives in the terminal, integrates deeply with git, supports multi-agent orchestration, and operates fully offline with local models. It aims to rival Claude Code and OpenCode in capability while giving users full control over models, tools, and data.

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
