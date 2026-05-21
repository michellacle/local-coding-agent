"""Tool Registry — tool schemas, registration, execution, and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolSchema:
    """Schema for a single tool.

    Attributes:
        name: Unique tool name.
        description: Human-readable description for the LLM.
        parameters: JSON Schema object describing the tool's parameters.
    """

    name: str
    description: str
    parameters: dict


class ToolRegistry:
    """Registry that maps tool names to schemas and implementations.

    Provides input validation against JSON schema before calling
    the underlying function.
    """

    def __init__(self) -> None:
        self._schemas: dict[str, ToolSchema] = {}
        self._functions: dict[str, Callable] = {}

    def register(self, schema: ToolSchema, fn: Callable) -> None:
        """Register a tool with its schema and implementation.

        Args:
            schema: The tool's name, description, and parameter schema.
            fn: The callable to invoke when the tool is executed.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if schema.name in self._schemas:
            raise ValueError(f"Tool '{schema.name}' already registered")
        self._schemas[schema.name] = schema
        self._functions[schema.name] = fn

    def list_tools(self) -> list[ToolSchema]:
        """Return all registered tool schemas."""
        return list(self._schemas.values())

    def execute(self, name: str, args: dict) -> Any:
        """Validate and execute a tool by name.

        Args:
            name: Registered tool name.
            args: Arguments to pass to the tool.

        Returns:
            The tool's return value.

        Raises:
            KeyError: If tool name is not registered.
            ValueError: If validation fails.
        """
        if name not in self._schemas:
            raise KeyError(f"Tool '{name}' not found")

        schema = self._schemas[name]
        self._validate(schema, args)

        fn = self._functions[name]
        return fn(**args)

    def get_definitions(self) -> str:
        """Return formatted tool definitions for the LLM system prompt.

        Returns a text block each tool's name, description, and parameters.
        """
        if not self._schemas:
            return "No tools available."

        lines: list[str] = []
        lines.append("Available tools:")
        lines.append("")
        for schema in self._schemas.values():
            lines.append(f"- {schema.name}: {schema.description}")
            if schema.parameters:
                params_desc = ", ".join(
                    f"{k} ({v.get('type', 'any')})"
                    for k, v in schema.parameters.get("properties", {}).items()
                )
                if params_desc:
                    lines.append(f"  Parameters: {params_desc}")
            lines.append("")
        return "\n".join(lines)

    # -- validation helpers --

    @staticmethod
    def _validate(schema: ToolSchema, args: dict) -> None:
        """Validate args against a ToolSchema's JSON parameters."""
        props = schema.parameters.get("properties", {})
        required = schema.parameters.get("required", [])

        # Check required params
        for req in required:
            if req not in args:
                raise ValueError(
                    f"Tool '{schema.name}' missing required parameter '{req}'"
                )

        # Check types
        type_map = {"string": str, "number": (int, float), "integer": int, "boolean": bool}
        for param_name, value in args.items():
            if param_name in props:
                expected_type_name = props[param_name].get("type")
                if expected_type_name and expected_type_name in type_map:
                    expected = type_map[expected_type_name]
                    if not isinstance(value, expected):
                        raise ValueError(
                            f"Parameter '{param_name}' expected type "
                            f"'{expected_type_name}', got '{type(value).__name__}'"
                        )
