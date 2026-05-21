# Product Requirements Document: Local Coding Agent

**Version:** 1.0  
**Date:** May 21, 2026  
**Status:** Draft  
**Source Material:**
- "Agentic Architectural Patterns for Building Multi-Agent Systems" by Dr. Ali Arsanjani & Juan Pablo Bustos (Packt, 2026)
- Competitive analysis of 15+ coding agents (thinksmart.life)

---

## 1. Product Vision

Build a **local-first, provider-agnostic coding agent** grounded in proven agentic architectural patterns. The agent lives in the terminal, integrates deeply with git, supports multi-agent orchestration, and can operate fully offline with local models. It should rival Claude Code and OpenCode in capability while giving the user full control over models, tools, and data.

**Core differentiators:**
- Fully local operation (no cloud dependency)
- Provider-agnostic with bring-your-own-model
- Pattern-driven architecture (each capability maps to a documented agentic pattern)
- Transparent reasoning (explanations for every action)
- Deep integration with developer workflows (git, terminal, IDE)

---

## 2. Competitive Landscape Summary

### Market Leaders (SWE-bench scores, May 2026)

| Rank | Agent | Score | Model | Interface |
|------|-------|-------|-------|-----------|
| 1 | Claude Code | 80.8% | Opus 4.6 | CLI |
| 2 | OpenCode | ~76% | BYO | CLI |
| 3 | Cursor | ~74% | Opus 4.5 | IDE |
| 4 | Windsurf | ~73% | Varied | IDE |

### Key Competitors and Their Positioning

**CLI-First:**
- **Claude Code** (Anthropic) — Market leader. 1M token context, deep git integration, multi-agent teams. Weaknesses: terminal-only, Anthropic-locked, expensive.
- **OpenCode** (OpenCode AI) — Open-source, BYO provider, LSP + MCP, headless server mode. Weaknesses: newer, smaller ecosystem.
- **OpenAI Codex** (OpenAI) — CLI agent with Codex model, git operations. Weaknesses: OpenAI-locked.
- **Aider** — Open-source, multi-model, git-aware. Weaknesses: limited multi-agent.

**IDE-Native:**
- **Cursor** — Deep IDE integration, multi-agent, great UX. Weaknesses: no CLI mode, closed ecosystem.
- **Windsurf** — Codeium-backed, IDE-focused. Weaknesses: limited CLI.
- **Zed** — Fast, built-in AI. Weaknesses: young agent capabilities.

**Cloud Platforms:**
- **Devin** (Cognition) — Fully autonomous, end-to-end planning. Weaknesses: cloud-only, expensive.
- **OpenHands** — Open-source cloud agent. Weaknesses: requires internet.

### Market Gaps Identified

1. **Truly local-first agents** — Most require cloud APIs. None work well fully offline.
2. **Provider-agnostic with deep git** — OpenCode is closest but less mature.
3. **Pattern-driven transparency** — No competitor exposes the agentic patterns they use.
4. **Cost control** — Local models + selective cloud fallback = predictable costs.

---

## 3. Agentic Patterns to Implement

Source: Arsanjani & Bustos, organized by development phase.

### Phase 1 Patterns (Foundation)

| Pattern | Problem Solved | Capability |
|---------|---------------|------------|
| LLM Selection & Deployment | Choose right model for task complexity and resource constraints | Model router with complexity-based routing |
| Function Calling | Agent needs to perform actions beyond text generation | Typed tool/function schemas with validation |
| RAG (Retrieval-Augmented Generation) | Agent needs domain-specific context not in training data | Vector-indexed knowledge base retrieval |

### Phase 2 Patterns (Core Capabilities)

| Pattern | Problem Solved | Capability |
|---------|---------------|------------|
| Multi-Agent Coordination | Complex tasks require parallel specialization | Orchestrator-delegate architecture |
| Agent Protocol (MCP) | Agents need standardized tool discovery and calling | MCP server/client protocol support |
| Agent-to-Agent (A2A) Communication | Agents need to share context and coordinate | Structured inter-agent messaging |
| Adaptive Retry | LLM failures need graceful recovery | Context-aware retry with strategy escalation |
| Agent Calls Human | Some decisions require human judgment | Human-in-the-loop interruption and approval |

### Phase 3 Patterns (Safety & Reliability)

| Pattern | Problem Solved | Capability |
|---------|---------------|------------|
| Authentication & Authorization | Agents access sensitive resources | Per-action auth controls and scopes |
| Explainability & Compliance | Users need to trust agent decisions | Action explanations and audit trails |
| Adversarial Testing & Red Teaming | Agents face malicious inputs or prompts | Input sanitization and prompt injection detection |
| In-Context Learning (ICL) | Agent improves within a session without fine-tuning | Dynamic prompt adaptation from feedback |
| Agent Fine-Tuning | Long-term capability enhancement | Optional model fine-tuning pipeline |

---

## 4. Functional Requirements

### 4.1 Terminal Interface (FR-001)

**MoSCoW: Must Have**

The agent operates primarily in the terminal with a conversational interface.

**Requirements:**
- Accept natural language instructions as input
- Stream responses token-by-token for low-latency feedback
- Support inline command execution with visible output
- Handle multi-turn conversations with persistent session context
- Support keyboard shortcuts (cancel, interrupt, history navigation)
- Render structured output (code diffs, tables, file trees) with syntax highlighting

**Competitive baseline:** Claude Code, OpenCode

### 4.2 Codebase Understanding (FR-002)

**MoSCoW: Must Have**

The agent must understand the project structure and codebase context.

**Requirements:**
- Auto-detect project type (language, framework, build system)
- Build a file index for fast codebase-wide search
- Parse file contents on demand with configurable depth limits
- Support ripgrep-style content search with regex
- Understand code dependencies (imports, references, call graphs)
- Load project-specific context files (AGENTS.md, CLAUDE.md, SKILL.md)
- Respect .gitignore and .ignore patterns

**Competitive baseline:** Claude Code (auto-summarization), Cursor (codebase-wide)

### 4.3 File Operations (FR-003)

**MoSCoW: Must Have**

The agent must read, write, and edit files safely.

**Requirements:**
- Read files with pagination and line numbers
- Write files (create new, overwrite existing)
- Perform targeted find-and-replace edits with fuzzy matching
- Validate syntax after edits (Python, JSON, YAML, TOML)
- Show unified diffs for all changes
- Support multi-file patches in a single operation
- Create parent directories automatically
- Enforce file size limits and safety guards

**Competitive baseline:** All major agents

### 4.4 Terminal/Shell Execution (FR-004)

**MoSCoW: Must Have**

The agent must execute shell commands for builds, installs, testing, etc.

**Requirements:**
- Execute foreground commands with configurable timeouts
- Execute background commands with progress tracking
- Support process management (list, poll, kill, wait)
- Run commands in specified working directories
- Capture and display stdout/stderr
- Support PTY mode for interactive CLI tools
- Auto-detect command success/failure from exit codes
- Sandboxing option for untrusted commands

**Competitive baseline:** Claude Code, OpenCode

### 4.5 Git Integration (FR-005)

**MoSCoW: Must Have**

Deep git integration mirroring Claude Code's capabilities.

**Requirements:**
- Initialize repositories, configure remotes
- Create branches for feature work
- Stage, commit, and push changes with meaningful messages
- Generate and apply diffs/patches
- Create pull requests with descriptions
- View commit history and log
- Resolve merge conflicts with AI assistance
- Run pre-commit hooks and quality gates
- Auto-generate commit messages from changes

**Competitive baseline:** Claude Code (deepest), OpenCode

### 4.6 Multi-Agent Orchestration (FR-006)

**MoSCoW: Must Have**

Spawn subagents for parallel or specialized tasks.

**Requirements:**
- Delegate tasks to isolated subagent sessions
- Support batch parallel execution (up to N concurrent)
- Each subagent gets its own context, terminal, and toolset
- Subagents return structured summaries
- Support orchestrator agents that can spawn their own workers
- Configurable nesting depth (default: 1)
- Token usage tracking per agent
- Subagent cancellation and timeout

**Competitive baseline:** Claude Code (Agent Teams), Cursor

### 4.7 LLM Provider Routing (FR-007)

**MoSCoW: Must Have**

Route requests to the appropriate model based on task complexity.

**Requirements:**
- Support multiple providers (Ollama, OpenAI, Anthropic, custom)
- Provider-agnostic interface (BYO model)
- Complexity-based routing (simple tasks -> fast/small models)
- Fallback chains (primary model fails -> try next)
- Per-task model override capability
- Local-first default (Ollama/llama.cpp first)
- Connection health monitoring
- Token usage tracking and cost estimation

**Competitive baseline:** OpenCode (BYO), Cursor (multi-model)

### 4.8 Retrieval-Augmented Generation (FR-008)

**MoSCoW: Must Have**

Enrich agent responses with relevant context from knowledge bases.

**Requirements:**
- Index local documents (PDFs, markdown, code docs)
- Vector similarity search over indexed content
- Embedding model selection (local or cloud)
- Auto-index project documentation
- Session search across conversation history
- Skill/procedural memory retrieval
- Configurable context window size

**Competitive baseline:** Claude Code (auto-summarization), Cursor

### 4.9 Function Calling / Tool Use (FR-009)

**MoSCoW: Must Have**

Typed function schemas that the agent can call reliably.

**Requirements:**
- Define tool schemas with parameters, types, and descriptions
- Validate inputs before execution
- Return structured outputs for downstream processing
- Support conditional tool loading (reduce token overhead)
- MCP-compatible tool definitions
- Error handling with retry strategies
- Tool discovery and listing

**Competitive baseline:** All major agents

### 4.10 Human-in-the-Loop (FR-010)

**MoSCoW: Must Have**

Interrupt agent flow for human decisions when needed.

**Requirements:**
- Agent asks for confirmation on destructive operations
- Multi-choice prompts for decision points
- Open-ended clarification questions
- Approval workflows for code changes
- Interrupt and resume capability
- Escalation on repeated failures
- User preference learning from decisions

**Competitive baseline:** Claude Code, OpenCode, Codex

### 4.11 Persistent Memory (FR-011)

**MoSCoW: Should Have**

Save durable facts that survive across sessions.

**Requirements:**
- User preference memory (communication style, habits, conventions)
- Environment/memory notes (facts, quirks, lessons learned)
- Add/replace/remove operations
- Memory injection into future sessions
- Compact, human-readable format
- Size limits and pruning policies
- Distinction between user profile and agent notes

**Competitive baseline:** Claude Code (limited), Hermes Agent

### 4.12 Skill Management (FR-012)

**MoSCoW: Should Have**

Procedural memory for reusable workflows.

**Requirements:**
- Create/update/delete skills (SKILL.md format)
- Skill discovery and auto-loading based on task context
- Categorized skill library
- Supporting files (references, templates, scripts)
- Skill versioning
- In-repo SKILL.md for project-specific workflows
- Auto-save suggestion after complex tasks

**Competitive baseline:** Hermes Agent (unique)

### 4.13 Adaptive Retry (FR-013)

**MoSCoW: Should Have**

Graceful recovery from failures with escalating strategies.

**Requirements:**
- Retry with different parameters on tool call failures
- Switch models on repeated errors
- Decompose tasks on complex failures
- Log failure patterns for learning
- Exponential backoff for rate limits
- Context-aware retry (different approach, not just repeat)

**Competitive baseline:** Claude Code, OpenCode

### 4.14 Explainability (FR-014)

**MoSCoW: Should Have**

Transparent reasoning for every action.

**Requirements:**
- Explain why a specific tool was chosen
- Show reasoning before destructive operations
- Provide audit trail of all actions in a session
- Confidence scores for recommendations
- Alternative approach suggestions
- Action summaries at session end

**Competitive baseline:** No strong competitor here (differentiator)

### 4.15 Cron Jobs / Scheduled Tasks (FR-015)

**MoSCoW: Could Have**

Schedule recurring agent tasks.

**Requirements:**
- Create/list/update/pause/resume/remove scheduled jobs
- Cron-style scheduling (every N minutes, hourly, daily)
- One-shot delayed tasks
- Job output delivery to user channels
- Script-based jobs (skip LLM for simple tasks)
- Chained jobs (output of A feeds into B)
- Token usage tracking per job

**Competitive baseline:** Hermes Agent (unique)

### 4.16 Media Generation (FR-016)

**MoSCoW: Could Have**

Generate images, audio, and other media.

**Requirements:**
- Text-to-image generation
- Text-to-speech with voice selection
- Audio playback and transcription
- Image analysis with vision models
- Video generation capability

**Competitive baseline:** Not a coding agent feature, but nice-to-have

### 4.17 Browser Automation (FR-017)

**MoSCoW: Could Have**

Navigate and interact with web pages.

**Requirements:**
- Navigate to URLs
- Click elements, fill forms, type text
- Take screenshots and analyze with vision AI
- Read page content (accessibility tree)
- Scroll, navigate back, press keys
- Console log capture and JavaScript evaluation
- Bot detection evasion for public sites

**Competitive baseline:** Devin, OpenHands

### 4.18 MCP Server Support (FR-018)

**MoSCoW: Should Have**

Connect to external MCP servers for extended capabilities.

**Requirements:**
- Connect to stdio and HTTP MCP servers
- Auto-discover and register tools from MCP servers
- Configure servers in config file
- Call MCP tools seamlessly alongside native tools
- Handle MCP server errors and reconnections

**Competitive baseline:** OpenCode, Claude Code

### 4.19 Adversarial Protection (FR-019)

**MoSCoW: Should Have**

Protect against prompt injection and malicious inputs.

**Requirements:**
- Detect prompt injection attempts in retrieved content
- Sanitize untrusted inputs before processing
- Rate limiting on tool calls
- Command allowlisting for sensitive operations
- Warning on potentially destructive actions

**Competitive baseline:** Claude Code (partial), OpenCode

---

## 5. Non-Functional Requirements

### 5.1 Performance (NFR-001)
- First token latency: < 500ms for local models, < 2s for cloud
- File read operations: < 100ms for files under 10KB
- Search operations: < 500ms for codebases under 100K LOC
- Tool call execution: < 2s for simple operations

### 5.2 Reliability (NFR-002)
- No data loss on crash (session persistence)
- Graceful degradation when models are unavailable
- Automatic recovery from transient network failures
- State checkpointing for long-running tasks

### 5.3 Security (NFR-003)
- No data exfiltration to cloud by default
- API keys stored securely (env vars, never in code)
- File system sandboxing option
- Command execution confirmation for destructive ops
- Audit log of all agent actions

### 5.4 Extensibility (NFR-004)
- Plugin architecture for new tool categories
- Custom command definitions
- Configurable via YAML
- Project-specific configuration (.agent/config)
- Skill system for sharing workflows

### 5.5 Observability (NFR-005)
- Token usage tracking per session and per provider
- Cost estimation for cloud providers
- Session logging and replay
- Performance metrics (latency, throughput)
- Error tracking and reporting

---

## 6. System Architecture

### Three-Layer Architecture (from Arsanjani & Bustos)

```
+--------------------------------------------------+
|              PRESENTATION LAYER                  |
|  Terminal UI | CLI Commands | Conversational     |
+--------------------------------------------------+
|               ORCHESTRATION LAYER                |
|  Agent Core | Task Planning | Multi-Agent Coord  |
|  Memory     | Skill Loader  | Human-in-the-Loop  |
+--------------------------------------------------+
|               INFRASTRUCTURE LAYER               |
|  Model Router | Tool Registry | Vector Store     |
|  File System  | Terminal      | Git Client       |
|  MCP Servers  | Embedding API | Browser Engine   |
+--------------------------------------------------+
```

### Component Map

| Component | Responsibility | Layer |
|-----------|---------------|-------|
| TerminalUI | Conversational interface, streaming | Presentation |
| AgentCore | Main agent loop, context management | Orchestration |
| TaskPlanner | Decompose goals into steps | Orchestration |
| MultiAgent | Subagent spawning and coordination | Orchestration |
| MemoryStore | Persistent memory (user + notes) | Orchestration |
| SkillLoader | Discover and load skills | Orchestration |
| HumanLoop | Clarification and approval prompts | Orchestration |
| ModelRouter | Route tasks to appropriate models | Infrastructure |
| ToolRegistry | Register and call tools | Infrastructure |
| VectorStore | Embedding and similarity search | Infrastructure |
| FileSystem | File read/write/patch operations | Infrastructure |
| TerminalExec | Shell command execution | Infrastructure |
| GitClient | Git operations | Infrastructure |
| MCPServer | MCP protocol implementation | Infrastructure |
| BrowserEngine | Web page interaction | Infrastructure |

---

## 7. Recommended Tech Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Language | Python 3.12+ | Rich ecosystem for ML/LLM tooling |
| CLI Framework | Rich / Textual | Terminal UI with streaming, tables, syntax highlighting |
| LLM Client | OpenAI-compatible SDK | Provider-agnostic, works with Ollama/vLLM |
| Local Inference | Ollama / llama.cpp | Local-first model serving |
| Embeddings | Local (nomic-embed-text) or Ollama | No cloud dependency |
| Vector Store | Chroma / FAISS | Lightweight, local, no external deps |
| File Search | ripgrep (via subprocess) | Fast, regex, already available |
| Git | git / gh CLI | Native integration |
| PDF Extraction | pymupdf / pymupdf4llm | Fast, reliable, local |
| Config | YAML (config.yaml) | Human-readable, widely understood |
| Testing | pytest | Standard Python test framework |
| Packaging | uv / pip | Python package management |
| Browser | Playwright | Cross-platform browser automation |

---

## 8. Phased Build Plan

### Phase 1: Foundation (Weeks 1-3)

**Goal:** Working agent that can read files, run commands, and call an LLM.

| Week | Deliverable | Patterns |
|------|------------|----------|
| 1 | Model router + LLM client | LLM Selection & Deployment |
| 1 | Terminal UI with streaming | - |
| 2 | File operations (read/write/patch) | - |
| 2 | Terminal execution | - |
| 2 | Function calling framework | Function Calling |
| 3 | Git integration | - |
| 3 | RAG over local docs | RAG |

**Milestone:** Agent can answer questions about a codebase, read files, run commands, and make edits.

### Phase 2: Core Capabilities (Weeks 4-6)

**Goal:** Multi-agent orchestration, skills, memory, and human-in-the-loop.

| Week | Deliverable | Patterns |
|------|------------|----------|
| 4 | Multi-agent delegation | Multi-Agent Coordination, A2A |
| 4 | Persistent memory | - |
| 5 | Skill system | - |
| 5 | Human-in-the-loop | Agent Calls Human |
| 5 | Adaptive retry | Adaptive Retry |
| 6 | MCP server support | Agent Protocol (MCP) |
| 6 | Search (session + web) | RAG |

**Milestone:** Agent can delegate to subagents, remember context across sessions, and learn workflows as skills.

### Phase 3: Safety & Polish (Weeks 7-8)

**Goal:** Production-ready agent with safety, explainability, and extensibility.

| Week | Deliverable | Patterns |
|------|------------|----------|
| 7 | Explainability & audit trail | Explainability & Compliance |
| 7 | Adversarial protection | Adversarial Testing & Red Teaming |
| 8 | Browser automation | - |
| 8 | Cron jobs / scheduling | - |
| 8 | In-context learning | In-Context Learning |
| 8 | Config and extension system | Authentication & Authorization |

**Milestone:** Production-ready coding agent competitive with Claude Code and OpenCode.

---

## 9. Success Criteria

### Quantitative

| Metric | Target |
|--------|--------|
| SWE-bench score | >= 70% (competitive baseline) |
| First token latency (local) | < 500ms |
| File operation latency | < 100ms |
| Tool call success rate | >= 95% |
| Session memory recall accuracy | >= 90% |
| Subagent task completion rate | >= 85% |

### Qualitative

- User can work fully offline with local models
- Every action is explained and auditable
- Agent learns user preferences over time
- Skills system enables sharing best practices
- No cloud dependency for core functionality
- Developer experience matches or exceeds Claude Code

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Local models underperform on complex tasks | High | Model router falls back to cloud models |
| Context window overflow on large codebases | Medium | Smart indexing, summarization, depth limits |
| Multi-agent coordination complexity | Medium | Start with depth-1, validate before nesting |
| Prompt injection via retrieved content | High | Input sanitization, content isolation |
| Tool call failures cascade | Medium | Adaptive retry with strategy escalation |
| Performance regression as features add up | Medium | Benchmark suite, performance budgets |

---

## 11. Open Questions

1. Should the agent support IDE extensions (VS Code, Neovim) or stay CLI-first?
2. What is the minimum viable local model for acceptable coding quality?
3. How should we handle project-specific skills vs. global skills?
4. Should cron jobs be part of v1 or deferred?
5. What is the target deployment model (pip package, standalone binary, Docker)?
