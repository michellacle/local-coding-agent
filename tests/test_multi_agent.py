"""Tests for multi-agent delegation — spawn subagents with isolated contexts."""

from unittest.mock import MagicMock, patch
from concurrent.futures import Future

import pytest

from local_agent.multi_agent import (
    AVAILABLE_TOOLSETS,
    BatchResult,
    DelegateAgent,
    DelegationResult,
    delegate_batch,
    delegate_task,
)


class TestDelegationResult:
    """Test DelegationResult dataclass."""

    def test_defaults(self):
        r = DelegationResult(task_id=1, goal="test", summary="done")
        assert r.status == "completed"
        assert r.error == ""
        assert r.tool_calls_made == 0

    def test_error_result(self):
        r = DelegationResult(task_id=2, goal="fail", summary="", status="error", error="boom")
        assert r.status == "error"
        assert r.error == "boom"


class TestBatchResult:
    """Test BatchResult dataclass."""

    def test_all_succeeded(self):
        results = [
            DelegationResult(task_id=1, goal="a", summary="ok"),
            DelegationResult(task_id=2, goal="b", summary="ok"),
        ]
        br = BatchResult(results=results, total_tasks=2, successful=2, failed=0)
        assert br.all_succeeded is True
        assert br.total_tasks == 2

    def test_some_failed(self):
        results = [
            DelegationResult(task_id=1, goal="a", summary="ok"),
            DelegationResult(task_id=2, goal="b", summary="", status="error", error="fail"),
        ]
        br = BatchResult(results=results, total_tasks=2, successful=1, failed=1)
        assert br.all_succeeded is False

    def test_all_failed(self):
        results = [
            DelegationResult(task_id=1, goal="a", summary="", status="error", error="fail"),
        ]
        br = BatchResult(results=results, total_tasks=1, successful=0, failed=1)
        assert br.all_succeeded is False


class TestAvailableToolsets:
    """Test AVAILABLE_TOOLSETS constants."""

    def test_file_toolset(self):
        assert "read_file" in AVAILABLE_TOOLSETS["file"]
        assert "write_file" in AVAILABLE_TOOLSETS["file"]

    def test_terminal_toolset(self):
        assert "execute_command" in AVAILABLE_TOOLSETS["terminal"]

    def test_git_toolset(self):
        assert "git_init" in AVAILABLE_TOOLSETS["git"]
        assert "git_commit" in AVAILABLE_TOOLSETS["git"]

    def test_search_toolset(self):
        assert "search_files" in AVAILABLE_TOOLSETS["search"]
        assert "list_directory" in AVAILABLE_TOOLSETS["search"]


class TestDelegateAgentInit:
    """Test DelegateAgent initialization."""

    def setup_method(self) -> None:
        self.router = MagicMock()
        self.router.config = MagicMock()
        self.router.config.host = "localhost"
        self.router.config.port = 7778
        self.router.config.model = "test-model"
        self.router.config.deterministic = False
        self.registry = MagicMock()
        self.registry._schemas = {}
        self.registry._functions = {}

    def test_default_concurrent(self):
        da = DelegateAgent(router=self.router, parent_registry=self.registry)
        assert da._max_concurrent == 3

    def test_custom_concurrent(self):
        da = DelegateAgent(router=self.router, parent_registry=self.registry, max_concurrent=2)
        assert da._max_concurrent == 2

    def test_concurrent_clamped_max(self):
        da = DelegateAgent(router=self.router, parent_registry=self.registry, max_concurrent=10)
        assert da._max_concurrent == 3

    def test_concurrent_clamped_min(self):
        da = DelegateAgent(router=self.router, parent_registry=self.registry, max_concurrent=0)
        assert da._max_concurrent == 1

    def test_default_spawn_depth(self):
        da = DelegateAgent(router=self.router, parent_registry=self.registry)
        assert da._max_spawn_depth == 1

    def test_default_child_turns(self):
        da = DelegateAgent(router=self.router, parent_registry=self.registry)
        assert da._max_child_turns == 8


class TestCloneRouter:
    """Test router cloning."""

    def setup_method(self) -> None:
        self.router = MagicMock()
        self.router.config = MagicMock()
        self.router.config.host = "testhost"
        self.router.config.port = 9999
        self.router.config.model = "custom-model"
        self.router.config.deterministic = True
        self.registry = MagicMock()
        self.registry._schemas = {}
        self.registry._functions = {}

    def test_clone_preserves_config(self):
        da = DelegateAgent(router=self.router, parent_registry=self.registry)
        child = da._clone_router()
        assert child.config.host == "testhost"
        assert child.config.port == 9999
        assert child.config.model == "custom-model"
        assert child.config.deterministic is True


class TestBuildChildRegistry:
    """Test child registry building with toolset filtering."""

    def setup_method(self) -> None:
        self.router = MagicMock()
        self.router.config = MagicMock()
        self.router.config.host = "localhost"
        self.router.config.port = 7778
        self.router.config.model = "test"
        self.router.config.deterministic = False

        from local_agent.tool_registry import ToolRegistry, ToolSchema

        self.registry = ToolRegistry()
        # Register a few tools
        self.registry.register(
            schema=ToolSchema(name="read_file", description="Read", parameters={"type": "object", "properties": {}, "required": []}),
            fn=lambda **kw: "content",
        )
        self.registry.register(
            schema=ToolSchema(name="execute_command", description="Exec", parameters={"type": "object", "properties": {}, "required": []}),
            fn=lambda **kw: "output",
        )
        self.registry.register(
            schema=ToolSchema(name="git_init", description="Git", parameters={"type": "object", "properties": {}, "required": []}),
            fn=lambda **kw: "ok",
        )

    def test_inherit_all_when_none(self):
        da = DelegateAgent(router=self.router, parent_registry=self.registry)
        child = da._build_child_registry(None)
        assert "read_file" in child._schemas
        assert "execute_command" in child._schemas
        assert "git_init" in child._schemas

    def test_filter_single_toolset(self):
        da = DelegateAgent(router=self.router, parent_registry=self.registry)
        child = da._build_child_registry(["file"])
        assert "read_file" in child._schemas
        assert "execute_command" not in child._schemas

    def test_filter_multiple_toolsets(self):
        da = DelegateAgent(router=self.router, parent_registry=self.registry)
        child = da._build_child_registry(["file", "terminal"])
        assert "read_file" in child._schemas
        assert "execute_command" in child._schemas
        assert "git_init" not in child._schemas

    def test_unknown_toolset_ignored(self):
        da = DelegateAgent(router=self.router, parent_registry=self.registry)
        child = da._build_child_registry(["nonexistent"])
        assert len(child._schemas) == 0


class TestBuildSystemPrompt:
    """Test child system prompt generation."""

    def setup_method(self) -> None:
        self.router = MagicMock()
        self.router.config = MagicMock()
        self.router.config.host = "localhost"
        self.router.config.port = 7778
        self.router.config.model = "test"
        self.router.config.deterministic = False
        self.registry = MagicMock()
        self.registry._schemas = {}
        self.registry._functions = {}

    def test_prompt_includes_goal(self):
        da = DelegateAgent(router=self.router, parent_registry=self.registry)
        child_reg = MagicMock()
        child_reg.get_definitions.return_value = "TOOLS: none"
        prompt = da._build_system_prompt(child_reg, "Goal: fix bug X")
        assert "fix bug X" in prompt

    def test_prompt_includes_sub_agent_identity(self):
        da = DelegateAgent(router=self.router, parent_registry=self.registry)
        child_reg = MagicMock()
        child_reg.get_definitions.return_value = ""
        prompt = da._build_system_prompt(child_reg, "some context")
        assert "sub-agent" in prompt.lower()


class TestDelegateToolFunction:
    """Test the tool-level delegation functions."""

    def test_delegate_no_delegator(self):
        result = delegate_task(goal="test", context="info")
        assert "error" in result
        assert "not configured" in result["error"].lower()

    def test_delegate_batch_no_delegator(self):
        result = delegate_batch(tasks=[{"goal": "test"}])
        assert "error" in result


class TestRunChildAgent:
    """Test child agent execution."""

    def setup_method(self) -> None:
        self.router = MagicMock()
        self.router.config = MagicMock()
        self.router.config.host = "localhost"
        self.router.config.port = 7778
        self.router.config.model = "test"
        self.router.config.deterministic = False

        from local_agent.tool_registry import ToolRegistry

        self.registry = ToolRegistry()

    @patch("local_agent.multi_agent.AgentCore")
    @patch("local_agent.multi_agent.ModelRouter")
    def test_successful_delegation(self, mock_router_cls, mock_agent_cls):
        mock_router_instance = MagicMock()
        mock_router_cls.return_value = mock_router_instance
        mock_agent_instance = MagicMock()
        mock_agent_instance.run_turn.return_value = "Task completed: file created."
        mock_agent_instance._history = []
        mock_agent_cls.return_value = mock_agent_instance

        da = DelegateAgent(router=self.router, parent_registry=self.registry)
        result = da.delegate(goal="Create a file", context="In /tmp")

        assert result.status == "completed"
        assert result.summary == "Task completed: file created."
        assert result.tool_calls_made == 0

    @patch("local_agent.multi_agent.AgentCore")
    @patch("local_agent.multi_agent.ModelRouter")
    def test_delegation_with_tool_calls(self, mock_router_cls, mock_agent_cls):
        mock_router_instance = MagicMock()
        mock_router_cls.return_value = mock_router_instance
        mock_agent_instance = MagicMock()
        mock_agent_instance.run_turn.return_value = "Done."
        mock_agent_instance._history = [
            {"role": "user", "content": "Tool result: file content"},
            {"role": "user", "content": "Tool result: command output"},
        ]
        mock_agent_cls.return_value = mock_agent_instance

        da = DelegateAgent(router=self.router, parent_registry=self.registry)
        result = da.delegate(goal="Do something", context="")

        assert result.status == "completed"
        assert result.tool_calls_made == 2

    @patch("local_agent.multi_agent.AgentCore")
    @patch("local_agent.multi_agent.ModelRouter")
    def test_delegation_error(self, mock_router_cls, mock_agent_cls):
        mock_router_instance = MagicMock()
        mock_router_cls.return_value = mock_router_instance
        mock_agent_instance = MagicMock()
        mock_agent_instance.run_turn.side_effect = Exception("Connection failed")
        mock_agent_cls.return_value = mock_agent_instance

        da = DelegateAgent(router=self.router, parent_registry=self.registry)
        result = da.delegate(goal="Do something", context="")

        assert result.status == "error"
        assert "Connection failed" in result.error


class TestBatchDelegation:
    """Test parallel batch delegation."""

    def setup_method(self) -> None:
        self.router = MagicMock()
        self.router.config = MagicMock()
        self.router.config.host = "localhost"
        self.router.config.port = 7778
        self.router.config.model = "test"
        self.router.config.deterministic = False

        from local_agent.tool_registry import ToolRegistry

        self.registry = ToolRegistry()

    @patch("local_agent.multi_agent.AgentCore")
    @patch("local_agent.multi_agent.ModelRouter")
    def test_batch_all_succeed(self, mock_router_cls, mock_agent_cls):
        mock_router_instance = MagicMock()
        mock_router_cls.return_value = mock_router_instance

        def make_agent(*args, **kwargs):
            agent = MagicMock()
            agent.run_turn.return_value = "Done."
            agent._history = []
            return agent

        mock_agent_cls.side_effect = make_agent

        da = DelegateAgent(router=self.router, parent_registry=self.registry)
        tasks = [
            {"goal": "Task A", "context": "Context A"},
            {"goal": "Task B", "context": "Context B"},
        ]
        result = da.delegate_batch(tasks)

        assert result.successful == 2
        assert result.failed == 0
        assert result.all_succeeded is True
        assert len(result.results) == 2

    @patch("local_agent.multi_agent.AgentCore")
    @patch("local_agent.multi_agent.ModelRouter")
    def test_batch_some_fail(self, mock_router_cls, mock_agent_cls):
        mock_router_instance = MagicMock()
        mock_router_cls.return_value = mock_router_instance

        call_count = [0]

        def make_agent(*args, **kwargs):
            call_count[0] += 1
            agent = MagicMock()
            if call_count[0] == 1:
                agent.run_turn.return_value = "Done."
                agent._history = []
            else:
                agent.run_turn.side_effect = Exception("Fail")
            return agent

        mock_agent_cls.side_effect = make_agent

        da = DelegateAgent(router=self.router, parent_registry=self.registry)
        tasks = [
            {"goal": "Task A"},
            {"goal": "Task B"},
        ]
        result = da.delegate_batch(tasks)

        assert result.total_tasks == 2
        assert result.successful == 1
        assert result.failed == 1
        assert result.all_succeeded is False

    def test_batch_respects_max_concurrent(self):
        da = DelegateAgent(
            router=self.router, parent_registry=self.registry, max_concurrent=2
        )
        assert da._max_concurrent == 2

    def test_task_counter_increments(self):
        da = DelegateAgent(router=self.router, parent_registry=self.registry)
        assert da._task_counter == 0
        with patch.object(da, "_run_child_agent", return_value=DelegationResult(
            task_id=1, goal="test", summary="ok"
        )):
            da.delegate(goal="test")
        assert da._task_counter == 1
