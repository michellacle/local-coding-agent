# Phase 6 Implementation Plan: Gap Closure + Benchmarking

**Date:** May 22, 2026
**Status:** Draft
**Based on:** PRODUCT_REQUIREMENTS.md v1.0 gap analysis + SWE-bench / benchmarking research

---

## Part A — Gap Closure (Finishing Phase 1-5 Features)

These are features claimed as "done" in PLAN.md but have incomplete implementations.

### A1. Git Workflow Completion (FR-005)

**Missing:**
- PR creation (via `gh` CLI or GitHub REST API)
- Auto-generated commit messages from diff analysis
- AI-assisted merge conflict resolution

**Implementation:**

1. **`git_create_pr(path, title, body, base, head)`** — Wraps `gh pr create` with title/body. Falls back to REST API if `gh` not installed. Returns PR URL.

2. **`git_commit_auto_message(path)`** — Runs `git diff --cached`, sends diff to LLM via ModelRouter with prompt to generate a conventional-commit message, then calls `git_commit`. No user input needed — fully autonomous.

3. **`git_resolve_conflicts(path, strategy)`** — When `git_merge` returns conflict status, this tool reads conflicted files (with `<<<<<<<` markers), sends them to LLM with instructions to produce resolved versions, writes them, stages, and completes the merge commit. Strategy options: "ours", "theirs", "llm".

**Files:** `git_tools.py` (+150 lines), `__main__.py` (register 3 new tools), `test_git_tools.py` (+80 tests)

### A2. Codebase Understanding (FR-002)

**Missing:**
- Auto-detect project type (language, framework, build system)
- Dependency graph (imports, references, call graphs)

**Implementation:**

1. **`project_detection.py`** — New module. Scans for `pyproject.toml`, `package.json`, `Cargo.toml`, `go.mod`, `CMakeLists.txt`, etc. Returns a structured `ProjectInfo` dataclass: language, framework, build system, test runner, package manager. Heuristic-based, no LLM needed.

2. **`dependency_graph.py`** — New module. For Python: parses AST `import` statements. For JS: parses `require`/`import`. Returns a dict mapping file -> [dependencies]. Used by AgentCore to inject relevant context (if file A imports B, load B when working on A).

**Files:** `project_detection.py` (new, ~200 lines), `dependency_graph.py` (new, ~250 lines), `agent_core.py` (inject project detection into system prompt), `test_project_detection.py` (+60 tests), `test_dependency_graph.py` (+40 tests)

### A3. Sandbox Mode (NFR-003)

**Missing:** File system sandboxing for untrusted commands.

**Implementation:**

1. Add `sandbox_dir` parameter to `execute_command` in `terminal_tools.py`. When set, commands run with `chroot` or restricted to a temp directory via `workdir`. Write-only mode: commands can only create/modify files within sandbox.

2. Add `--sandbox` CLI flag in `__main__.py` that enables sandboxed mode globally.

**Files:** `terminal_tools.py` (+50 lines), `__main__.py` (+10 lines), `test_terminal_tools.py` (+15 tests)

### A4. Performance Baselines (NFR-001)

**Missing:** Benchmark suite for file ops, search ops, tool call latency.

**Implementation:** See Part C below (benchmarking infrastructure).

---

## Part B — Benchmarking Infrastructure

**Goal:** Run Local Coding Agent against established benchmarks (SWE-bench, HumanEval, MBPP) and get comparable scores to Claude Code (80.8%), OpenCode (~76%), Cursor (~74%).

### B1. What We Need to Change in the Software

The current agent is designed for interactive terminal use. To make it benchmarkable, we need:

**1. Headless "task runner" mode**

Current `__main__.py` has `--prompt` for one-shot commands, but benchmarks need:
- Feed a batch of N tasks from a JSONL file
- For each task: initialize a fresh AgentCore (clean history), run it, capture output
- Parse the agent's final filesystem state or code output
- Compare against expected answers
- Write results to structured output (JSON)

**New module:** `benchmark_runner.py`

```python
# Pseudocode structure
class BenchmarkRunner:
    def __init__(self, config: LLMConfig, benchmark: str):
        ...
    
    def run_task(self, task: BenchmarkTask) -> BenchmarkResult:
        # 1. Spin up isolated sandbox (temp dir, fresh git repo)
        # 2. Set up task context (seed repo, apply patch, etc.)
        # 3. Create AgentCore with task-specific system prompt
        # 4. Run agent_turn with task description
        # 5. Capture all tool calls, file changes, terminal output
        # 6. Evaluate result against expected answer
        # 7. Return structured result with metrics
        ...

    def run_batch(self, tasks: list[BenchmarkTask]) -> list[BenchmarkResult]:
        ...
```

**2. Deterministic execution hooks**

For reproducible benchmark runs:
- `--seed` flag to set temperature=0 and seed=0 on ModelRouter (already partially supported via `config.deterministic`)
- Log ALL LLM requests/responses to a JSONL trace file for replay/debugging
- Capture full timing: first-token latency, total response time, tool call count

**3. AgentCore observability**

Add hooks to AgentCore for benchmarking:
- `on_tool_call(tool_name, args)` callback
- `on_llm_request(messages)` callback
- `on_llm_response(response, latency_ms, tokens)` callback
- `on_turn_complete(turn_num, total_tools_used)` callback

These are NO-OPS by default (zero overhead in normal mode) but get wired up by BenchmarkRunner.

**4. Evaluation adapters**

Each benchmark has a different evaluation protocol:
- **SWE-bench:** Apply agent's generated patch to repo, run test suite, check pass/fail
- **HumanEval:** Extract generated function, run unit tests from dataset
- **MBPP:** Same as HumanEval but simpler tasks

**New module:** `benchmark_eval.py` with `SWE BenchEvaluator`, `HumanEvalEvaluator`, `MBPPEvaluator`.

### B2. Target Benchmarks

| Benchmark | What It Tests | How Many Tasks | Our Strategy |
|-----------|--------------|----------------|--------------|
| **HumanEval** | Generate function from docstring | 164 | Easiest entry point. Agent receives docstring, writes Python file, we run the tests. |
| **MBPP** | Solve programming puzzles | 399 | Similar to HumanEval, slightly harder. Good second step. |
| **SWE-bench (lite)** | Fix real GitHub issues | 300 | Full repo context. Agent reads codebase, diagnoses, patches, runs tests. This is the real differentiator. |
| **Aider-defined** | Custom regression tests | TBD | Internal benchmarks for specific features (git workflow, multi-file edits, etc.) |

**Recommendation:** Start with HumanEval (B2.1) as a proof of concept, then MBPP, then SWE-bench lite. SWE-bench is the hardest and most valuable score.

### B3. Software Changes Required

**New modules:**
- `benchmark_runner.py` — Headless task runner, sandbox management, trace capture
- `benchmark_eval.py` — Evaluation adapters for each benchmark
- `benchmark_prompts.py` — Task-specific system prompts (HumanEval vs SWE-bench need different framing)
- `sandbox.py` — Isolated environment: temp dir, fresh git, restricted FS access

**Modified modules:**
- `agent_core.py` — Add observability callbacks (on_tool_call, on_llm_request, etc.)
- `model_router.py` — Add request/response logging to trace file
- `__main__.py` — Add `--benchmark` mode: `python -m local_agent --benchmark humaneval --tasks data/humaneval.jsonl --output results.json`

**Infrastructure:**
- `scripts/download_humaneval.py` — Fetch HumanEval dataset
- `scripts/download_mbpp.py` — Fetch MBPP dataset
- `scripts/download_swe_bench_lite.py` — Fetch SWE-bench lite
- `data/` directory for benchmark datasets (gitignored)

**Estimated effort:**
- New modules: ~1,200 lines
- Modifications: ~300 lines
- Tests: ~400 tests

---

## Part C — Test Quality Improvements

Current tests are 719 passing, but a significant portion are shallow mocks. Before benchmarking, we need to harden the test suite.

### C1. Integration Test Gaps

Current `test_integration.py` has 12 tests but they mostly mock the LLM. We need real integration paths:

1. **Full agent loop integration** — AgentCore + real ToolRegistry + mocked LLM that returns realistic tool calls. Verify the agent actually chains tools correctly (read file -> analyze -> write file -> commit).

2. **Git workflow end-to-end** — Create temp repo -> agent edits files -> agent stages -> agent commits with auto-message -> verify git log.

3. **Multi-agent delegation** — Parent agent delegates to 2 subagents, collects results, produces final output. Verify isolation (subagent A can't see subagent B's state).

### C2. Property-Based Testing

Add `hypothesis` library for fuzzing:
- Feed random JSON to tool call parser, verify it either parses or gives a clear error (no crashes)
- Feed random file paths to read_file/patch_file, verify safety guards trigger
- Feed random cron expressions, verify parser handles them gracefully

### C3. Benchmark-Specific Regression Tests

Once we run HumanEval/MBPP, take a few tasks the agent gets right and wrong:
- Save the task + trace as regression tests
- Re-run after every change to ensure scores don't silently regress

---

## Part D — Execution Order

### Sprint 1: Gap Closure (2-3 days)
1. A1 — Git workflow completion (PR creation, auto-commit, conflict resolution)
2. A2 — Project detection + dependency graph
3. A3 — Sandbox mode for execute_command

### Sprint 2: Test Hardening (1-2 days)
4. C1 — Integration test gaps
5. C2 — Property-based testing for parsers
6. A4 — Performance baseline measurements

### Sprint 3: Benchmarking Infrastructure (3-4 days)
7. B3 — AgentCore observability hooks + trace logging
8. B1 — BenchmarkRunner + Sandbox + headless mode
9. B2.1 — HumanEval adapter + evaluation
10. Run first HumanEval benchmark, report score

### Sprint 4: Expand Benchmarks (3-4 days)
11. B2.2 — MBPP adapter
12. B2.3 — SWE-bench lite adapter (the big one)
13. C3 — Regression tests from benchmark results

---

## Part E — Success Metrics

| Metric | Target |
|--------|--------|
| HumanEval pass@1 | >= 40% (baseline for local models) |
| MBPP pass@1 | >= 35% |
| SWE-bench lite resolved | >= 15% |
| Gap closure — all FR-005 git features complete | Yes |
| Integration tests with real tool chaining | >= 10 |
| Benchmark trace replay works | Yes |

---

## Part F — Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Local models (qwen3.5:9b Q4) underperform badly on HumanEval | Start with the larger model on moneymaker (27B). Fall back to cloud via ModelRouter if needed. |
| SWE-bench repos are huge and don't fit in context | Project detection + dependency graph (A2) lets us load only relevant files. Smart context windowing. |
| Benchmark runs are slow (each task = full LLM call + tool chain) | Parallel execution across tasks via multi_agent. Cache LLM responses for identical prompts. |
| Our JSON-based tool calling format doesn't match what benchmarks expect | BenchmarkRunner translates between benchmark format and our internal tool protocol. |
