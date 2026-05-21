# Implementation Plan: Local Coding Agent

**Based on:** PRODUCT_REQUIREMENTS.md v1.0
**Date:** May 21, 2026
**Status:** Active

---

## Current State Assessment

### What's Built (Phase 1 - Partial)

| Component | File | FR Coverage | Status |
|-----------|------|-------------|--------|
| ModelRouter | model_router.py | FR-007 (partial) | Done - Ollama support, streaming, 120s timeout |
| ToolRegistry | tool_registry.py | FR-009 (core) | Done - schemas, validation, execution |
| FileTools | tools/file_tools.py | FR-003 | Done - read, write, patch |
| TerminalTools | tools/terminal_tools.py | FR-004 (partial) | Done - foreground commands only |
| GitTools | tools/git_tools.py | FR-005 (partial) | Done - init, add, commit, status, diff |
| AgentCore | agent_core.py | Core loop | Partial - single turn, basic tool parsing |
| TerminalUI | terminal_ui.py | FR-001 (partial) | Done - input/output, no streaming |
| Config | config.py | NFR-004 | Done - env-based LLMConfig + AppConfig |
| Non-interactive mode | __main__.py | CLI | Done - --prompt flag |
| Integration tests | tests/test_integration.py | End-to-end | Done - 2 passing tests |

### What's Missing - Ordered by Priority

---

## Phase 1 Completion (Foundation)

### 1. Multi-turn Tool Chaining
**FRs:** Core agent loop, FR-009
**Problem:** Agent stops after one tool call. Real tasks need 2-5 tool calls in sequence (read file, patch file, run test, commit).
**Scope:**
- AgentCore: loop until no tool call detected, with max-turns safety limit
- After tool execution, feed result back to LLM for next decision
- Track tool call chain in history for context
**Files:** `src/local_agent/agent_core.py`
**Tests:** `tests/test_agent_core.py` - multi-turn tool chain

### 2. Codebase Understanding Tools
**FRs:** FR-002
**Problem:** Agent can't search codebases or list directories. Needs ripgrep-style search and file discovery.
**Scope:**
- `search_files` tool: regex content search + file name glob search (via ripgrep subprocess)
- `list_directory` tool: recursive directory listing with depth limit
- Respect .gitignore patterns
**Files:** `src/local_agent/tools/search_tools.py`
**Tests:** `tests/test_search_tools.py`

### 3. Streaming Terminal UI
**FRs:** FR-001
**Problem:** User waits for full response. Streaming tokens gives immediate feedback.
**Scope:**
- Wire AgentCore streaming mode through TerminalUI
- Token-by-token rendering with Rich console
- Preserve tool call detection in streaming mode
**Files:** `src/local_agent/terminal_ui.py`, `src/local_agent/agent_core.py`

### 4. Background Terminal Execution
**FRs:** FR-004
**Problem:** Long-running commands (builds, tests) block the agent.
**Scope:**
- `execute_command` with `background` flag
- Process management: list, poll, wait, kill
- Session ID tracking for background processes
**Files:** `src/local_agent/tools/terminal_tools.py`
**Tests:** `tests/test_terminal_tools.py`

### 5. RAG over Local Docs
**FRs:** FR-008
**Problem:** Agent has no project context. Needs to index and retrieve from local docs (README, PRD, code comments).
**Scope:**
- Document indexer: chunk markdown/text/PDF files
- Embedding API via Ollama (nomic-embed-text)
- Vector store: lightweight in-memory or Chroma local
- `retrieve_context` tool: semantic search over indexed docs
- Auto-index project root on startup
**Files:** `src/local_agent/vector_store.py`, `src/local_agent/document_indexer.py`
**Tests:** `tests/test_vector_store.py`, `tests/test_document_indexer.py`

### 6. Git Workflow Deepening
**FRs:** FR-005
**Problem:** Current git tools are basic (init, add, commit, status, diff). Need branch management, push, log.
**Scope:**
- `git_branch`: create and checkout branches
- `git_push`: push to remote
- `git_log`: commit history
- `git_merge`: merge with conflict detection
**Files:** `src/local_agent/tools/git_tools.py`

---

## Phase 2 (Core Capabilities)

### 7. Multi-Agent Orchestration
**FRs:** FR-006
**Problem:** Complex tasks need parallel specialization (e.g., research A and B simultaneously).
**Scope:**
- `DelegateAgent`: spawn child agent with isolated session
- Single-task and batch (up to 3 parallel) modes
- Child gets its own context, terminal, toolset subset
- Structured summary returns
- Configurable nesting depth (default: 1)
**Files:** `src/local_agent/multi_agent.py`
**Tests:** `tests/test_multi_agent.py`

### 8. Persistent Memory
**FRs:** FR-011
**Problem:** Agent forgets everything between sessions.
**Scope:**
- Two stores: user profile + agent notes
- Add/replace/remove operations
- Human-readable format (declarative facts)
- Size limits with pruning
- Auto-inject into system prompt on startup
**Files:** `src/local_agent/memory.py`
**Tests:** `tests/test_memory.py`

### 9. Skill System
**FRs:** FR-012
**Problem:** No reusable workflows. Complex procedures repeated each session.
**Scope:**
- SKILL.md format (YAML frontmatter + markdown body)
- Create/update/delete/list skills
- Categorized skill library in `~/.local-coding-agent/skills/`
- In-repo SKILL.md for project-specific workflows
- Auto-load based on task context matching
- Supporting files (references, templates, scripts)
**Files:** `src/local_agent/skill_manager.py`, `src/local_agent/skill_loader.py`
**Tests:** `tests/test_skill_manager.py`

### 10. Human-in-the-Loop
**FRs:** FR-010
**Problem:** Agent makes decisions that need human judgment.
**Scope:**
- `clarify` tool: multi-choice and open-ended prompts
- Confirmation on destructive operations (rm, git force push)
- Interrupt and resume capability
- Approval workflow for code changes
**Files:** `src/local_agent/human_loop.py`
**Tests:** `tests/test_human_loop.py`

### 11. Adaptive Retry
**FRs:** FR-013
**Problem:** Tool calls and LLM responses fail. Need graceful recovery.
**Scope:**
- Retry with different parameters on tool failures
- Switch models on repeated errors
- Exponential backoff for rate limits
- Context-aware retry (different approach, not blind repeat)
- Max retry limit with escalation to human
**Files:** `src/local_agent/agent_core.py` (retry logic), `src/local_agent/retry.py`
**Tests:** `tests/test_retry.py`

### 12. MCP Server Support
**FRs:** FR-018
**Problem:** External tools not available natively. MCP standardizes tool discovery.
**Scope:**
- Connect to stdio and HTTP MCP servers
- Auto-discover and register MCP tools
- Config file for MCP server definitions
- Seamless MCP tool calls alongside native tools
**Files:** `src/local_agent/mcp_client.py`
**Tests:** `tests/test_mcp_client.py`

---

## Phase 3 (Safety & Polish)

### 13. Explainability & Audit Trail
**FRs:** FR-014
**Scope:**
- Action explanations before tool execution
- Session audit log (all actions, timestamps, results)
- Confidence scores for recommendations
- Session summary at end
**Files:** `src/local_agent/explainability.py`

### 14. Adversarial Protection
**FRs:** FR-019
**Scope:**
- Prompt injection detection in retrieved content
- Input sanitization
- Command allowlisting for sensitive ops
- Rate limiting on tool calls
**Files:** `src/local_agent/safety.py`

### 15. Browser Automation
**FRs:** FR-017
**Scope:**
- Playwright-based browser engine
- Navigate, click, type, screenshot
- Accessibility tree extraction
- Console log capture
**Files:** `src/local_agent/browser_engine.py`

### 16. Cron Jobs / Scheduled Tasks
**FRs:** FR-015
**Scope:**
- Create/list/update/pause/resume/remove scheduled jobs
- Cron-style scheduling
- Script-based jobs (skip LLM)
- Output delivery
**Files:** `src/local_agent/cron_scheduler.py`

### 17. Configuration System
**FRs:** NFR-004
**Scope:**
- YAML config file (`~/.local-coding-agent/config.yaml`)
- Project-specific config (`.agent/config.yaml`)
- Provider definitions with fallback chains
- Tool enable/disable toggles
**Files:** `src/local_agent/config.py` (expand)

---

## Execution Order

Recommended implementation sequence (dependencies first):

1. Multi-turn tool chaining (unblocks everything)
2. Search tools (codebase understanding)
3. Streaming UI (better UX)
4. Background terminal (long commands)
5. RAG system (context awareness)
6. Git deepening (workflow completeness)
7. Persistent memory (cross-session)
8. Skill system (reusable workflows)
9. Human-in-the-loop (safety)
10. Multi-agent (parallelism)
11. Adaptive retry (reliability)
12. MCP support (extensibility)
13-17. Phase 3 items (polish)

---

## Integration Test Strategy

Each major feature addition should have:
1. Unit tests for the module
2. Integration test validating end-to-end flow
3. Updated smoke test in `tests/test_integration.py`

Run after every feature: `./scripts/run_integration.sh`
