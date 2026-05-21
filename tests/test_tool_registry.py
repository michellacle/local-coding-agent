"""Tests for ToolRegistry — tool schemas, registration, execution, validation."""

import pytest

from local_agent.tool_registry import ToolSchema, ToolRegistry


class TestToolSchema:
    """Test ToolSchema dataclass."""

    def test_create_schema(self):
        schema = ToolSchema(
            name="greet",
            description="Say hello",
            parameters={"type": "object", "properties": {"name": {"type": "string"}}},
        )
        assert schema.name == "greet"
        assert schema.description == "Say hello"
        assert schema.parameters["type"] == "object"

    def test_schema_repr(self):
        schema = ToolSchema(name="test", description="desc", parameters={})
        assert schema.name in repr(schema)


class TestToolRegistryRegister:
    """Test tool registration."""

    def test_register_adds_tool(self):
        registry = ToolRegistry()
        schema = ToolSchema(
            name="add",
            description="Add two numbers",
            parameters={"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
        )
        registry.register(schema, lambda a, b: a + b)
        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "add"

    def test_register_multiple_tools(self):
        registry = ToolRegistry()
        registry.register(ToolSchema(name="a", description="A", parameters={}), lambda: 1)
        registry.register(ToolSchema(name="b", description="B", parameters={}), lambda: 2)
        assert len(registry.list_tools()) == 2

    def test_register_duplicate_raises(self):
        registry = ToolRegistry()
        registry.register(ToolSchema(name="x", description="X", parameters={}), lambda: 1)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(ToolSchema(name="x", description="X2", parameters={}), lambda: 2)


class TestToolRegistryExecute:
    """Test tool execution."""

    def test_execute_calls_function(self):
        registry = ToolRegistry()
        registry.register(
            ToolSchema(name="add", description="Add", parameters={
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            }),
            lambda a, b: a + b,
        )
        result = registry.execute("add", {"a": 3, "b": 4})
        assert result == 7

    def test_execute_no_args(self):
        registry = ToolRegistry()
        registry.register(ToolSchema(name="ping", description="Ping", parameters={}), lambda: "pong")
        result = registry.execute("ping", {})
        assert result == "pong"

    def test_execute_unknown_tool_raises(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.execute("nonexistent", {})

    def test_execute_missing_required_param_raises(self):
        registry = ToolRegistry()
        registry.register(
            ToolSchema(name="say", description="Say", parameters={
                "type": "object",
                "properties": {"msg": {"type": "string"}},
                "required": ["msg"],
            }),
            lambda msg: msg,
        )
        with pytest.raises(ValueError, match="missing required"):
            registry.execute("say", {})

    def test_execute_type_mismatch_raises(self):
        registry = ToolRegistry()
        registry.register(
            ToolSchema(name="double", description="Double", parameters={
                "type": "object",
                "properties": {"n": {"type": "number"}},
                "required": ["n"],
            }),
            lambda n: n * 2,
        )
        with pytest.raises(ValueError, match="type"):
            registry.execute("double", {"n": "not_a_number"})


class TestToolRegistryDefinitions:
    """Test get_definitions output."""

    def test_get_definitions_formats_output(self):
        registry = ToolRegistry()
        registry.register(ToolSchema(name="add", description="Add two numbers", parameters={}), lambda: None)
        registry.register(ToolSchema(name="mul", description="Multiply", parameters={}), lambda: None)
        definitions = registry.get_definitions()
        assert "add" in definitions
        assert "mul" in definitions
        assert "Add two numbers" in definitions

    def test_empty_definitions(self):
        registry = ToolRegistry()
        definitions = registry.get_definitions()
        assert "No tools available" in definitions
