"""Multi-agent delegation — spawn subagents with isolated contexts.

Supports:
- Single-task delegation: one child agent with a specific goal and context
- Batch (parallel) delegation: up to 3 child agents running concurrently
- Each child gets its own conversation history, tool subset, and terminal session
- Structured summary returns (self-reports from each child)
- Configurable nesting depth (default: 1, leaf-only)

Dependencies injected for testability:
    - router_factory: callable that creates a ModelRouter for each child
    - registry_factory: callable that creates a ToolRegistry for each child
    - max_concurrent: max parallel children (default: 3)
    - max_spawn_depth: nesting depth limit (default: 1)
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_agent.agent_core import AgentCore
from local_agent.model_router import ModelRouter
from local_agent.tool_registry import ToolRegistry, ToolSchema


# Available toolset names that can be selectively enabled
AVAILABLE_TOOLSETS = {
    "file": ["read_file", "write_file", "patch_file"],
    "terminal": ["execute_command"],
    "git": ["git_init", "git_add", "git_commit", "git_status", "git_diff"],
    "search": ["search_files", "list_directory"],
    "human_loop": ["clarify", "confirm"],
    "memory": ["memory", "memory_search", "memory_list"],
    "skill": ["skill_manage", "skill_view", "skills_list"],
}


@dataclass
class DelegationResult:
    """Result from a single delegated task."""

    task_id: int
    goal: str
    summary: str
    status: str = "completed"  # completed, error, timeout, cancelled
    error: str = ""
    tool_calls_made: int = 0


@dataclass
class BatchResult:
    """Combined result from a batch of delegated tasks."""

    results: list[DelegationResult]
    total_tasks: int
    successful: int
    failed: int

    @property
    def all_succeeded(self) -> bool:
        return self.failed == 0


class DelegateAgent:
    """Spawn and coordinate child agents for delegated tasks.

    Args:
        router: Parent ModelRouter (cloned for children).
        parent_registry: Parent ToolRegistry (filtered for children).
        max_concurrent: Max parallel child agents (default: 3).
        max_spawn_depth: Max nesting depth (default: 1 for leaf-only).
        max_child_turns: Max turns per child agent (default: 8).
    """

    def __init__(
        self,
        router: ModelRouter,
        parent_registry: ToolRegistry,
        max_concurrent: int = 3,
        max_spawn_depth: int = 1,
        max_child_turns: int = 8,
    ) -> None:
        self._router = router
        self._parent_registry = parent_registry
        self._max_concurrent = max(1, min(max_concurrent, 3))
        self._max_spawn_depth = max_spawn_depth
        self._max_child_turns = max_child_turns
        self._task_counter = 0
        self._lock = threading.Lock()

    def _clone_router(self) -> ModelRouter:
        """Clone the parent router for a child agent."""
        from local_agent.config import LLMConfig

        child_config = LLMConfig(
            host=self._router.config.host,
            port=self._router.config.port,
            model=self._router.config.model,
            deterministic=self._router.config.deterministic,
        )
        return ModelRouter(config=child_config)

    def _build_child_registry(
        self, toolsets: list[str] | None
    ) -> ToolRegistry:
        """Build a filtered tool registry for a child agent.

        Args:
            toolsets: List of toolset names to include. None means inherit all.

        Returns:
            A new ToolRegistry with only the requested toolsets.
        """
        registry = ToolRegistry()

        if toolsets is None:
            # Inherit all parent tools
            for schema in self._parent_registry._schemas.values():
                fn = self._parent_registry._functions[schema.name]
                registry.register(schema=schema, fn=fn)
            return registry

        # Selectively include toolsets
        for toolset_name in toolsets:
            if toolset_name in AVAILABLE_TOOLSETS:
                tool_names = AVAILABLE_TOOLSETS[toolset_name]
                for name in tool_names:
                    if name in self._parent_registry._schemas:
                        schema = self._parent_registry._schemas[name]
                        fn = self._parent_registry._functions[name]
                        registry.register(schema=schema, fn=fn)

        return registry

    def _build_system_prompt(self, registry: ToolRegistry, context: str) -> str:
        """Build a child agent system prompt with context."""
        base: str = (
            "You are a sub-agent delegated to handle a specific task.\n\n"
            "Task context:\n"
            f"{context}\n\n"
            "When you need to use a tool, respond with ONLY a JSON object containing "
            '"tool" (the tool name) and "args" (a dict of arguments). Do NOT include '
            "any text before or after the JSON. One tool call at a time — the result will "
            "be fed back to you so you can make the next call.\n\n"
            "When you have completed the task, respond with a concise summary of "
            "what you accomplished (or failed to do). Include any file paths, URLs, "
            "or identifiers that the parent agent can verify.\n\n"
        )
        tools_section: str = registry.get_definitions()
        return base + tools_section

    def _run_child_agent(
        self,
        task_id: int,
        goal: str,
        context: str,
        toolsets: list[str] | None,
    ) -> DelegationResult:
        """Run a single child agent for a delegated task.

        Args:
            task_id: Unique task identifier.
            goal: What the child should accomplish.
            context: Background information for the child.
            toolsets: Which toolsets to give the child.

        Returns:
            DelegationResult with the child's summary.
        """
        try:
            child_router = self._clone_router()
            child_registry = self._build_child_registry(toolsets)
            child_agent = AgentCore(
                router=child_router,
                registry=child_registry,
                streaming=False,
                max_turns=self._max_child_turns,
            )

            # Override system prompt builder to inject task context
            original_build = child_agent._build_system_prompt
            child_agent._build_system_prompt = lambda: self._build_system_prompt(  # type: ignore[assignment]
                child_registry, f"Goal: {goal}\n\n{context}"
            )

            summary = child_agent.run_turn(goal)

            # Count tool calls from history
            tool_calls = sum(
                1 for msg in child_agent._history
                if msg["role"] == "user" and msg["content"].startswith("Tool result:")
            )

            return DelegationResult(
                task_id=task_id,
                goal=goal,
                summary=summary,
                status="completed",
                tool_calls_made=tool_calls,
            )

        except Exception as e:
            return DelegationResult(
                task_id=task_id,
                goal=goal,
                summary="",
                status="error",
                error=str(e),
            )

    def delegate(
        self,
        goal: str,
        context: str = "",
        toolsets: list[str] | None = None,
    ) -> DelegationResult:
        """Delegate a single task to a child agent.

        Args:
            goal: What the child agent should accomplish.
            context: Background information (file paths, error messages, constraints).
            toolsets: Which toolsets to enable. None means inherit all.

        Returns:
            DelegationResult with the child's summary.
        """
        with self._lock:
            self._task_counter += 1
            task_id = self._task_counter

        return self._run_child_agent(task_id, goal, context, toolsets)

    def delegate_batch(
        self,
        tasks: list[dict[str, Any]],
    ) -> BatchResult:
        """Delegate multiple tasks to child agents running in parallel.

        Args:
            tasks: List of task dicts with keys:
                - goal (str, required): Task goal
                - context (str, optional): Task-specific context
                - toolsets (list[str], optional): Toolsets for this task

        Returns:
            BatchResult combining all child results.
        """
        num_tasks = min(len(tasks), self._max_concurrent)

        with self._lock:
            self._task_counter += num_tasks
            start_id = self._task_counter - num_tasks + 1

        futures = {}
        with ThreadPoolExecutor(max_workers=self._max_concurrent) as executor:
            for i, task in enumerate(tasks[:num_tasks]):
                task_id = start_id + i
                goal = task.get("goal", "")
                context = task.get("context", "")
                toolsets = task.get("toolsets")

                future = executor.submit(
                    self._run_child_agent, task_id, goal, context, toolsets
                )
                futures[future] = task_id

        results: list[DelegationResult] = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

        # Sort by task_id for deterministic order
        results.sort(key=lambda r: r.task_id)

        successful = sum(1 for r in results if r.status == "completed")
        failed = sum(1 for r in results if r.status != "completed")

        return BatchResult(
            results=results,
            total_tasks=len(results),
            successful=successful,
            failed=failed,
        )


def delegate_task(
    goal: str,
    context: str = "",
    toolsets: list[str] | None = None,
    _delegator: DelegateAgent | None = None,
) -> dict[str, Any]:
    """Tool function: delegate a task to a child agent.

    This is the tool interface that the LLM calls. It wraps DelegateAgent.delegate().

    Args:
        goal: What the child agent should accomplish.
        context: Background information for the child.
        toolsets: Which toolsets to enable (e.g., ["file", "terminal"]).
        _delegator: Internal — injected DelegateAgent instance.

    Returns:
        Dict with task_id, summary, status, error.
    """
    if _delegator is None:
        return {"error": "Delegation not configured. No parent DelegateAgent available."}

    result = _delegator.delegate(goal, context, toolsets)
    return {
        "task_id": result.task_id,
        "goal": result.goal,
        "summary": result.summary,
        "status": result.status,
        "error": result.error,
        "tool_calls_made": result.tool_calls_made,
    }


def delegate_batch(
    tasks: list[dict[str, Any]],
    _delegator: DelegateAgent | None = None,
) -> dict[str, Any]:
    """Tool function: delegate multiple tasks in parallel.

    Args:
        tasks: List of task dicts with goal, context, toolsets keys.
        _delegator: Internal — injected DelegateAgent instance.

    Returns:
        Dict with results, total_tasks, successful, failed.
    """
    if _delegator is None:
        return {"error": "Delegation not configured. No parent DelegateAgent available."}

    batch = _delegator.delegate_batch(tasks)
    return {
        "results": [
            {
                "task_id": r.task_id,
                "goal": r.goal,
                "summary": r.summary,
                "status": r.status,
                "error": r.error,
            }
            for r in batch.results
        ],
        "total_tasks": batch.total_tasks,
        "successful": batch.successful,
        "failed": batch.failed,
        "all_succeeded": batch.all_succeeded,
    }
