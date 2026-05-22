# Implementation Plan: Local Coding Agent

**Based on:** PRODUCT_REQUIREMENTS.md v1.0
**Date:** May 22, 2026
**Status:** Phase 1-4 Complete — 719 tests passing

---

## Current State Assessment

### What's Built

| # | Component | File(s) | Status | Tests |
|---|-----------|---------|--------|-------|
| 1 | ModelRouter | model_router.py | Done — Ollama support, streaming, 120s timeout | 16 |
| 2 | ToolRegistry | tool_registry.py | Done — schemas, validation, execution | 14 |
| 3 | FileTools | tools/file_tools.py | Done — read, write, patch | 24 |
| 4 | TerminalTools | tools/terminal_tools.py | Done — foreground + background, process mgmt | 19 |
| 5 | GitTools | tools/git_tools.py | Done — init, add, commit, status, diff, branch, push, log, merge | 18 |
| 6 | SearchTools | tools/search_tools.py | Done — ripgrep content/file search, directory listing | 15 |
| 7 | AgentCore | agent_core.py | Done — multi-turn tool chaining, max-turns safety | 17 |
| 8 | TerminalUI | terminal_ui.py | Done — streaming token rendering, Rich console | 14 |
| 9 | LLMConfig + AppConfig | config.py | Done — env vars, YAML, env interpolation, hot-reload | 28 |
| 10 | Non-interactive mode | __main__.py | Done — --prompt flag | 6 |
| 11 | Multi-Agent | multi_agent.py | Done — delegate_agent, batch (up to 3 parallel) | 12 |
| 12 | Memory | memory.py | Done — user profile + agent notes, CRUD, size limits | 10 |
| 25 | Task Planner | task_planner.py | Done — goal decomposition, complexity estimation, plan save/load | 28 |
| 26 | Complexity Routing | model_router.py | Done — routing rules, fallback chains, per-model stats | 18 |
| 27 | Token Tracking | model_router.py | Done — prompt/completion tokens, latency, error counts | 18 |
| 28 | Session Persistence | session_persist.py | Done — auto-save, restore on startup, session list | 15 |
| 29 | Slash Commands | terminal_ui.py | Done — /models, /plan, /stats | — |
| 13 | Skill Manager | skill_manager.py | Done — SKILL.md format, create/update/delete/list | 11 |
| 14 | Human-in-the-Loop | human_loop.py | Done — multi-choice/open-ended prompts | 10 |
| 15 | Adaptive Retry | retry.py | Done — exponential backoff, context-aware retry | 12 |
| 16 | RAG Pipeline | rag.py | Done — document indexer, vector store, Ollama/TF-IDF embeddings | 35 |
| 17 | Explainability | explainability.py | Done — audit trail, decision logging, self-assessment | 37 |
| 18 | Safety | safety.py | Done — prompt injection detection, command safety, rate limiting | 45 |
| 19 | Browser Engine | browser_engine.py | Done — Playwright navigate/click/type/screenshot/JS eval | 10 |
| 20 | Cron Scheduler | cron.py | Done — job management, cron parsing, persistence | 40 |
| 21 | Config System | config.py (ConfigManager) | Done — YAML, env interpolation, multi-source merge | 28 |
| 22 | Integration | test_integration.py | Done — end-to-end flows | 12 |
| 23 | MCP Client | mcp_client.py | Done — stdio + HTTP MCP server support | 12 |
| 24 | Vector Store (legacy) | vector_store.py | Done — cosine similarity, source filtering | 16 |

**Total tests: 719 passed**

---

## What's Missing - Ordered by Priority

---

## Phase 5 (Advanced IDE-like Features) — ALL DONE

### 30. LSP Client ✅
**Status:** Complete — JSON-RPC stdio LSP client: diagnostics, go-to-definition, references, symbols, hover info. Supports pyright, typescript-language-server, rust-analyzer, etc.

### 31. Project Context Auto-loading ✅
**Status:** Complete — Scans project for AGENTS.md, CLAUDE.md, .cursorrules, pyproject.toml, package.json, README.md, etc. Loads and formats them into system prompt blocks with priority ordering and size limits.

### 32. Diff Review Workflow ✅
**Status:** Complete — Proposed change management: propose changes, preview unified diffs, approve/reject individual files, batch apply/reject. Mirrors Claude Code's "apply" workflow.

### 33. In-Context Learning (ICL) ✅
**Status:** Complete — Tracks user corrections during a session, categorizes them (style/naming/logic/security), and injects few-shot examples into the system prompt. Corrections persist to JSON.

---

## Phase 1 Completion (Foundation) — ALL DONE

### 1. Multi-turn Tool Chaining ✅
**FRs:** Core agent loop, FR-009
**Status:** Complete — AgentCore loops until no tool call detected, with max-turns safety limit.

### 2. Codebase Understanding Tools ✅
**FRs:** FR-002
**Status:** Complete — `search_files` (regex content + file name glob), `list_directory` (recursive with depth limit), .gitignore support.

### 3. Streaming Terminal UI ✅
**FRs:** FR-001
**Status:** Complete — Token-by-token rendering with Rich console, streaming tool call detection.

### 4. Background Terminal Execution ✅
**FRs:** FR-004
**Status:** Complete — `execute_command` with `background` flag, process list/poll/wait/kill.

### 5. RAG over Local Docs ✅
**FRs:** FR-008
**Status:** Complete — Document indexer (chunk text/PDF), embedding API (Ollama, TF-IDF fallback), vector store (in-memory + SQLite persistence), semantic search.

### 6. Git Workflow Deepening ✅
**FRs:** FR-005
**Status:** Complete — Branch management, push, log, merge with conflict detection.

---

## Phase 2 (Core Capabilities) — ALL DONE

### 7. Multi-Agent Orchestration ✅
**FRs:** FR-006
**Status:** Complete — DelegateAgent with isolated sessions, batch mode (up to 3 parallel), structured summaries.

### 8. Persistent Memory ✅
**FRs:** FR-011
**Status:** Complete — Two stores (user profile + agent notes), CRUD, size limits, auto-inject into system prompt.

### 9. Skill System ✅
**FRs:** FR-012
**Status:** Complete — SKILL.md format, create/update/delete/list, categorized skill library, in-repo SKILL.md support.

### 10. Human-in-the-Loop ✅
**FRs:** FR-010
**Status:** Complete — Multi-choice and open-ended prompts, confirmation on destructive ops.

### 11. Adaptive Retry ✅
**FRs:** FR-013
**Status:** Complete — Exponential backoff, context-aware retry, max retry limit with escalation.

### 12. MCP Server Support ✅
**FRs:** FR-018
**Status:** Complete — stdio + HTTP MCP server connections, auto-discover and register tools.

---

## Phase 3 (Safety & Polish) — ALL DONE

### 13. Explainability & Audit Trail ✅
**FRs:** FR-014
**Status:** Complete — Decision logging, chain of thought, audit trail with JSON Lines persistence, self-assessment of outputs.

### 14. Adversarial Protection ✅
**FRs:** FR-019
**Status:** Complete — Prompt injection detection (12 patterns), command blocklist/allowlist, rate limiting, content sanitization.

### 15. Browser Automation ✅
**FRs:** FR-017
**Status:** Complete — Playwright-based browser engine: navigate, click, type, screenshot, accessibility tree, console capture, JS eval.

### 16. Cron Jobs / Scheduled Tasks ✅
**FRs:** FR-015
**Status:** Complete — Cron expression parsing, job CRUD, enable/disable, persistence, due job execution.

### 17. Configuration System ✅
**FRs:** NFR-004
**Status:** Complete — YAML config with multi-source merging, environment variable interpolation (`${VAR}`, `${VAR:-default}`), hot-reload with callbacks, schema validation, export.

---

## Execution Order (Completed)

1. ✅ Multi-turn tool chaining
2. ✅ Search tools
3. ✅ Streaming UI
4. ✅ Background terminal
5. ✅ RAG system
6. ✅ Git deepening
7. ✅ Persistent memory
8. ✅ Skill system
9. ✅ Human-in-the-loop
10. ✅ Multi-agent
11. ✅ Adaptive retry
12. ✅ MCP support
13. ✅ Explainability & audit trail
14. ✅ Adversarial protection
15. ✅ Browser automation
16. ✅ Cron jobs
17. ✅ Configuration system expansion

---

## Phase 4 (Advanced Features) — ALL DONE

### 25. Task Planner ✅
**Status:** Complete — Goal decomposition via LLM, per-step complexity estimation (simple/moderate/complex), plan persistence (save/load/list), `/plan` slash command.

### 26. Complexity-Based Routing + Fallback Chains ✅
**Status:** Complete — RoutingRule maps complexity tiers to model configs, fallback chains try alternate providers on failure, token/latency stats per model.

### 27. Session Persistence ✅
**Status:** Complete — Auto-save conversation history after each turn, restore on next launch, session metadata (model, turn count, timestamps).

### 28. Token Usage Tracking ✅
**Status:** Complete — Prompt/completion token counting, latency tracking, error counts. Viewable via `/stats` slash command.

---

## Integration Test Strategy

Each major feature addition should have:
1. Unit tests for the module
2. Integration test validating end-to-end flow
3. Updated smoke test in `tests/test_integration.py`

Run after every feature: `./scripts/run_integration.sh`
