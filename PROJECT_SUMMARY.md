# Local Coding Agent — Project Summary

## Your Code. Your Models. Your Machine.

**Local Coding Agent is a Claude Code rival that never sends your code to the cloud.** Built for privacy-conscious developers who want the power of AI pair programming without the monthly subscription or data leakage.

---

## The Problem

Every major AI coding tool has a catch:

- **Claude Code** ($20-200/mo) — sends everything to Anthropic. Locked into their ecosystem.
- **Cursor** ($20/mo) — closed-source. Your code leaves your machine.
- **OpenCode** — improving, but still maturing and less feature-complete.
- **Windsurf / Devin / OpenHands** — cloud-dependent. Your proprietary code becomes someone else's training data.

What if you could have the same capabilities — multi-agent orchestration, git integration, browser automation, RAG — but running entirely on your own hardware?

**That's what we built.**

---

## What Is It?

A terminal-based AI coding agent that rivals Claude Code in capability, runs fully offline, and costs nothing but your existing GPU.

**Ground floor specs:**
- Runs on an RTX 4070 (8GB) for small models, or dual RTX 3090s (48GB) for large ones
- Works with any OpenAI-compatible API — Ollama, vLLM, llama.cpp, or whatever you want
- Zero cloud dependencies. Ever.

---

## Built on Research, Not Hype

Every feature maps to a proven agentic architectural pattern from *"Agentic Architectural Patterns for Building Multi-Agent Systems"* by Dr. Ali Arsanjani & Juan Pablo Bustos (Packt, 2026). This isn't a random collection of features — it's a deliberate implementation of what academic research shows actually works.

| Pattern | Our Implementation |
|---------|-------------------|
| LLM Selection & Routing | Complexity-based model routing with fallback chains |
| Function Calling | Typed tool registry with 28+ tools and input validation |
| RAG | Vector-indexed knowledge base with Ollama embeddings |
| Multi-Agent Coordination | Parallel subagent delegation (up to 3 concurrent) |
| Agent Protocol (MCP) | Full stdio + HTTP MCP server support |
| Adaptive Retry | Context-aware retry with strategy escalation |
| Human-in-the-Loop | Multi-choice and open-ended approval prompts |
| Explainability | Audit trail with decision logging and self-assessment |
| Adversarial Protection | Prompt injection detection and command allowlisting |
| In-Context Learning | Learns from your corrections during a session |

---

## Capabilities

### Core Coding Workflow
- **Multi-turn tool chaining** — The agent loops through tool calls autonomously, chaining operations until the task is complete
- **Codebase understanding** — ripgrep-style content search, file indexing, dependency detection
- **File operations** — Read, write, and patch files with syntax validation and unified diffs
- **Diff review** — Preview all changes before they touch disk. Approve or reject individual files. No surprises.
- **Git integration** — Branch, commit, push, merge, log, stash, tag. Full git workflow from the agent.
- **LSP integration** — Real-time diagnostics, go-to-definition, symbol search, and references through language servers like pyright.

### Intelligence Layer
- **RAG over your docs** — Index PDFs, markdown, and code docs. The agent answers questions from your project's own documentation.
- **Project context** — Auto-loads AGENTS.md, CLAUDE.md, pyproject.toml, package.json, and more so the agent understands your project's conventions from the start.
- **In-context learning** — When you correct the agent, it remembers. Style preferences, naming conventions, logic fixes — all fed back into future responses in the same session.
- **Session persistence** — Crashes? No data loss. Sessions auto-save and restore on next launch.

### Multi-Agent Orchestration
- **Task delegation** — Spawn isolated subagents for parallel work. Up to 3 concurrent agents, each with their own context and tools.
- **Task planner** — Decompose complex goals into ordered step lists with complexity estimates. Simple tasks go to fast models, complex ones to powerful ones.
- **Complexity-based routing** — Automatic model selection based on task complexity, with fallback chains if a model fails.

### Safety & Transparency
- **Explainability** — Every action is logged with reasoning. Full audit trail in JSON format.
- **Adversarial protection** — Prompt injection detection (12+ patterns), command blocklisting, rate limiting.
- **Human-in-the-loop** — The agent asks before destructive operations. Multi-choice prompts for decisions that need your judgment.

### Beyond Code
- **Browser automation** — Navigate, click, fill forms, take screenshots, run JavaScript. Full Playwright integration.
- **Cron jobs** — Schedule recurring agent tasks. Daily code reviews, automated testing, monitoring.
- **MCP support** — Connect to external MCP servers for extended capabilities. Seamless tool discovery.
- **Persistent memory** — User profile and agent notes that survive across sessions. Skills system for reusable workflows.

### Developer Experience
- **28+ slash commands** — `/code`, `/plan`, `/stats`, `/provider`, `/models`, `/memory`, `/skill`, `/search`, `/git`, `/browser`, `/cron`, `/rag`, `/explain`, `/safety`, and more
- **Streaming output** — Token-by-token rendering so you see responses as they're generated
- **Terminal UI** — Rich console with syntax highlighting, diffs, tables, and file trees
- **Non-interactive mode** — `--prompt "do this"` for scripting and CI integration

---

## By the Numbers

| Metric | Value |
|--------|-------|
| Source files | 29 |
| Lines of code | ~12,000 |
| Test files | 30 |
| Passing tests | 850+ |
| Slash commands | 28+ |
| Agentic patterns implemented | 10 |
| Phases delivered | 5 |
| Git commits | 39 |

---

## Architecture

Three layers, inspired by the Arsanjani & Bustos framework:

```
PRESENTATION LAYER
  Terminal UI | Slash Commands | Streaming Output | Diff Review

ORCHESTRATION LAYER
  Agent Core | Task Planner | Multi-Agent | Memory | Skills
  Human-in-the-Loop | In-Context Learning | Session Persistence

INFRASTRUCTURE LAYER
  Model Router | Tool Registry | Vector Store | LSP Client
  File System | Terminal | Git | MCP | Browser | RAG
```

---

## Competitive Positioning

| Feature | Local Coding Agent | Claude Code | OpenCode |
|---------|-------------------|-------------|----------|
| Fully offline | Yes | No | Partial |
| Cost | $0 (your GPU) | $20-200/mo | $0 |
| Provider-agnostic | Yes | No | Yes |
| Multi-agent | Yes | Yes | Yes |
| Browser automation | Yes | No | No |
| Cron jobs | Yes | No | No |
| RAG over local docs | Yes | Auto-summarize | Limited |
| MCP support | Yes | Yes | Yes |
| LSP integration | Yes | No | Yes |
| Diff review | Yes | Yes | Partial |
| In-context learning | Yes | No | No |
| Explainability audit | Yes | Partial | No |
| Adversarial protection | Yes | Partial | Partial |

---

## Who Is This For?

- **Privacy-first developers** who won't let their proprietary code touch third-party servers
- **Cost-conscious engineers** tired of $200/month subscriptions
- **GPU owners** with an RTX 4070, 3090, A6000, or anything with 8-48GB VRAM
- **Teams building internal tools** who need an agent that works in air-gapped environments
- **Researchers** who want to understand agentic patterns in practice, not just theory

---

## How It Works

```bash
# Install
cd ~/.local-coding-agent
pip install -e .

# Run
python -m local_agent

# Or one-shot
python -m local_agent --prompt "refactor the auth module"
```

That's it. Point it at your project and start coding.

---

## Roadmap

**Phase 1-5:** Complete (foundation, core capabilities, safety, advanced features, IDE-like tools)

**Next:** Cloud provider fallback, fine-tuning pipeline, IDE plugin, collaborative multi-user mode, and performance optimizations targeting <500ms first-token latency.

---

## The Bottom Line

This is what a coding agent looks like when it's built on research instead of marketing. Every feature serves a proven pattern. Every line of code runs on your machine. Every dollar you'd spend on subscriptions stays in your pocket.

**Your code stays yours. Your models are yours. Your machine does the work.**

Built with ❤️ by developers, for developers who value sovereignty.
