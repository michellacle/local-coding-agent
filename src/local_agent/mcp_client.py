"""MCP Client — connect to external MCP servers and expose their tools.

Supports two transport modes:
  1. Stdio — spawn a subprocess, communicate via stdin/stdout JSON-RPC
  2. SSE  — connect to an HTTP SSE endpoint, POST JSON-RPC to a URL

Features:
  - Auto-discover tools via `tools/list`
  - Execute tools via `tools/call`
  - Register MCP tools into the agent's ToolRegistry seamlessly
  - Config-driven server definitions
  - Proper lifecycle (initialize → list_tools → call → notifications/keepalive)

Reference: https://modelcontextprotocol.io/specification
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON-RPC protocol
# ---------------------------------------------------------------------------


class JsonRpcError(Exception):
    """Raised on JSON-RPC error responses."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"JSON-RPC error {code}: {message}")


@dataclass
class JsonRpcRequest:
    """A JSON-RPC 2.0 request."""

    jsonrpc: str = "2.0"
    id: int = 0
    method: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
            "method": self.method,
        }
        if self.params:
            d["params"] = self.params
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class JsonRpcResponse:
    """A JSON-RPC 2.0 response."""

    jsonrpc: str = "2.0"
    id: int = 0
    result: Any = None
    error: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JsonRpcResponse:
        return cls(
            jsonrpc=d.get("jsonrpc", "2.0"),
            id=d.get("id", 0),
            result=d.get("result"),
            error=d.get("error"),
        )

    def raise_for_error(self) -> Any:
        if self.error:
            raise JsonRpcError(
                code=self.error.get("code", -1),
                message=self.error.get("message", "Unknown error"),
                data=self.error.get("data"),
            )
        return self.result


# ---------------------------------------------------------------------------
# Transport layer
# ---------------------------------------------------------------------------


@dataclass
class McpTransportConfig:
    """Configuration for an MCP server connection."""

    name: str
    transport: str  # "stdio" or "sse"
    command: str = ""           # for stdio: command to run
    args: list[str] = field(default_factory=list)  # for stdio: command args
    env: dict[str, str] = field(default_factory=dict)  # for stdio: extra env vars
    url: str = ""               # for SSE: SSE endpoint URL
    headers: dict[str, str] = field(default_factory=dict)  # for SSE: extra headers


class McpTransport(ABC):
    """Abstract MCP transport."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection."""

    @abstractmethod
    async def send(self, request: JsonRpcRequest) -> JsonRpcResponse:
        """Send a request and wait for the response."""

    @abstractmethod
    async def send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a one-way notification (no response expected)."""


class StdioTransport(McpTransport):
    """stdio transport: communicate with a subprocess over stdin/stdout."""

    def __init__(self, config: McpTransportConfig) -> None:
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._read_task: asyncio.Task | None = None
        self._next_id = 1
        self._pending: dict[int, asyncio.Future] = {}

    async def connect(self) -> None:
        env = dict(__import__("os").environ)
        env.update(self.config.env)

        self._process = await asyncio.create_subprocess_exec(
            self.config.command,
            *self.config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        self._read_task = asyncio.create_task(self._read_loop())
        logger.info("StdioTransport connected: %s", self.config.name)

    async def disconnect(self) -> None:
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass

        if self._process:
            if self._process.stdin:
                self._process.stdin.close()
                try:
                    await self._process.wait()
                except Exception:
                    pass
            if self._process.returncode is None:
                self._process.kill()
                try:
                    await self._process.wait()
                except Exception:
                    pass

        # Resolve any pending futures
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("Transport disconnected"))

        logger.info("StdioTransport disconnected: %s", self.config.name)

    async def _read_loop(self) -> None:
        if not self._process or not self._process.stdout:
            return

        buffer = b""
        try:
            while True:
                chunk = await self._process.stdout.readline()
                if not chunk:
                    break

                line_str = chunk.strip().decode("utf-8", errors="replace")
                if not line_str:
                    continue

                try:
                    d = json.loads(line_str)
                    if "id" in d:
                        # Response
                        resp = JsonRpcResponse.from_dict(d)
                        rid = resp.id
                        if rid in self._pending:
                            fut = self._pending.pop(rid)
                            if not fut.done():
                                fut.set_result(resp)
                        else:
                            logger.warning("No pending request for id=%s", rid)
                    elif "method" in d:
                        # Server notification (e.g., notifications/message)
                        logger.debug("Server notification: %s", d.get("method"))
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning("Failed to parse server message: %s (%s)", line_str, e)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("StdioTransport read error: %s", e)

    async def send(self, request: JsonRpcRequest) -> JsonRpcResponse:
        if not self._process or not self._process.stdin:
            raise RuntimeError("Transport not connected")

        rid = self._next_id
        self._next_id += 1
        request.id = rid

        fut: asyncio.Future[JsonRpcResponse] = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut

        msg = request.to_json() + "\n"
        self._process.stdin.write(msg.encode("utf-8"))
        await self._process.stdin.drain()

        # Wait for response with timeout
        try:
            resp = await asyncio.wait_for(fut, timeout=30.0)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise TimeoutError(f"MCP request timed out: {request.method}")

        return resp

    async def send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        if not self._process or not self._process.stdin:
            raise RuntimeError("Transport not connected")

        notification: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params:
            notification["params"] = params

        msg = json.dumps(notification) + "\n"
        self._process.stdin.write(msg.encode("utf-8"))
        await self._process.stdin.drain()


class SseTransport(McpTransport):
    """SSE transport: connect to an HTTP SSE server."""

    def __init__(self, config: McpTransportConfig) -> None:
        self.config = config
        self._next_id = 1
        self._pending: dict[int, asyncio.Future] = {}
        self._endpoint_url: str = ""
        self._sse_task: asyncio.Task | None = None
        import aiohttp
        self._session: aiohttp.ClientSession | None = None

    async def connect(self) -> None:
        import aiohttp

        self._session = aiohttp.ClientSession(headers=self.config.headers)

        # Connect to SSE stream to discover the messagePostEndpoint
        async with self._session.get(self.config.url) as resp:
            resp.raise_for_status()

            # Read SSE events to find "endpoint" event
            buffer = ""
            async for line in resp.content:
                line_str = line.decode("utf-8", errors="replace").rstrip("\n")
                if not line_str:
                    continue

                if line_str.startswith("data: "):
                    data = line_str[6:]
                    try:
                        event = json.loads(data)
                        if event.get("type") == "endpoint":
                            self._endpoint_url = event.get("url", "")
                            if self._endpoint_url:
                                break
                    except json.JSONDecodeError:
                        pass

        if not self._endpoint_url:
            raise RuntimeError(f"No endpoint discovered from SSE server at {self.config.url}")

        logger.info("SSETransport connected: %s (endpoint: %s)", self.config.name, self._endpoint_url)
        self._sse_task = asyncio.create_task(self._sse_read_loop())

    async def disconnect(self) -> None:
        if self._sse_task:
            self._sse_task.cancel()
            try:
                await self._sse_task
            except asyncio.CancelledError:
                pass

        if self._session:
            await self._session.close()
            self._session = None

        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("Transport disconnected"))

        logger.info("SSETransport disconnected: %s", self.config.name)

    async def _sse_read_loop(self) -> None:
        if not self._session:
            return

        try:
            async with self._session.get(self.config.url) as resp:
                resp.raise_for_status()
                async for line in resp.content:
                    line_str = line.decode("utf-8", errors="replace").rstrip("\n")
                    if not line_str:
                        continue

                    if line_str.startswith("data: "):
                        data = line_str[6:]
                        try:
                            d = json.loads(data)
                            if "id" in d:
                                resp_obj = JsonRpcResponse.from_dict(d)
                                rid = resp_obj.id
                                if rid in self._pending:
                                    fut = self._pending.pop(rid)
                                    if not fut.done():
                                        fut.set_result(resp_obj)
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning("Failed to parse SSE message: %s", e)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("SSETransport read error: %s", e)

    async def send(self, request: JsonRpcRequest) -> JsonRpcResponse:
        if not self._session or not self._endpoint_url:
            raise RuntimeError("Transport not connected")

        rid = self._next_id
        self._next_id += 1
        request.id = rid

        loop = asyncio.get_event_loop()
        fut: asyncio.Future[JsonRpcResponse] = loop.create_future()
        self._pending[rid] = fut

        try:
            async with self._session.post(
                self._endpoint_url,
                json=request.to_dict(),
                headers={"Content-Type": "application/json"},
            ) as resp:
                resp.raise_for_status()
        except Exception as e:
            self._pending.pop(rid, None)
            raise RuntimeError(f"SSE POST failed: {e}")

        try:
            return await asyncio.wait_for(fut, timeout=30.0)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise TimeoutError(f"MCP request timed out: {request.method}")

    async def send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        if not self._session or not self._endpoint_url:
            raise RuntimeError("Transport not connected")

        notification: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params:
            notification["params"] = params

        async with self._session.post(
            self._endpoint_url,
            json=notification,
            headers={"Content-Type": "application/json"},
        ) as resp:
            resp.raise_for_status()


# ---------------------------------------------------------------------------
# MCP Client
# ---------------------------------------------------------------------------


@dataclass
class McpTool:
    """A tool discovered from an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str  # which server it came from


class McpClient:
    """Client for connecting to MCP servers.

    Manages the lifecycle: connect → initialize → list tools → call tools.

    Args:
        config: MCP transport configuration.
    """

    def __init__(self, config: McpTransportConfig) -> None:
        self.config = config
        self.transport: McpTransport | None = None
        self._tools: list[McpTool] = []
        self._server_info: dict[str, Any] = {}
        self._capabilities: dict[str, Any] = {}

        # Create transport
        if config.transport == "stdio":
            self.transport = StdioTransport(config)
        elif config.transport == "sse":
            self.transport = SseTransport(config)
        else:
            raise ValueError(f"Unknown transport: {config.transport}")

    @property
    def is_connected(self) -> bool:
        return self.transport is not None

    async def connect(self) -> None:
        """Connect to the MCP server and initialize."""
        if not self.transport:
            raise RuntimeError("No transport configured")

        await self.transport.connect()

        # Initialize protocol
        init_request = JsonRpcRequest(
            method="initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {
                        "listChanged": True,
                    },
                },
                "clientInfo": {
                    "name": "local-coding-agent",
                    "version": "0.1.0",
                },
            },
        )

        resp = await self.transport.send(init_request)
        result = resp.raise_for_error()

        self._server_info = result.get("serverInfo", {})
        self._capabilities = result.get("capabilities", {})

        # Send initialized notification
        await self.transport.send_notification("notifications/initialized")

        # Discover tools
        await self.list_tools()

        logger.info(
            "MCPClient connected to %s: %d tools, server=%s",
            self.config.name,
            len(self._tools),
            self._server_info.get("name", "unknown"),
        )

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self.transport:
            await self.transport.disconnect()
            self.transport = None
            self._tools = []

    async def list_tools(self) -> list[McpTool]:
        """Discover available tools from the server."""
        if not self.transport:
            raise RuntimeError("Not connected")

        resp = await self.transport.send(JsonRpcRequest(method="tools/list"))
        result = resp.raise_for_error()

        tools = result.get("tools", [])
        self._tools = [
            McpTool(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
                server_name=self.config.name,
            )
            for t in tools
        ]

        return self._tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on the MCP server.

        Args:
            name: Tool name.
            arguments: Tool arguments.

        Returns:
            Tool result.
        """
        if not self.transport:
            raise RuntimeError("Not connected")

        resp = await self.transport.send(
            JsonRpcRequest(
                method="tools/call",
                params={
                    "name": name,
                    "arguments": arguments,
                },
            )
        )

        result = resp.raise_for_error()

        # MCP tool results come as a list of content blocks
        content_blocks = result.get("content", [])
        if not content_blocks:
            return result

        # Flatten to text
        texts: list[str] = []
        for block in content_blocks:
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif block.get("type") == "result":
                texts.append(json.dumps(block.get("result", "")))
            else:
                texts.append(json.dumps(block))

        return "\n".join(texts) if texts else json.dumps(result)


# ---------------------------------------------------------------------------
# MCP Manager — manages multiple server connections
# ---------------------------------------------------------------------------


class McpManager:
    """Manages connections to multiple MCP servers.

    Registers MCP tools into the agent's ToolRegistry, prefixing them
    with the server name to avoid collisions.

    Args:
        configs: List of MCP server configurations.
    """

    def __init__(self, configs: list[McpTransportConfig]) -> None:
        self.configs = configs
        self.clients: dict[str, McpClient] = {}

    async def connect_all(self) -> None:
        """Connect to all configured MCP servers."""
        for config in self.configs:
            client = McpClient(config)
            try:
                await client.connect()
                self.clients[config.name] = client
                logger.info("Connected to MCP server: %s (%d tools)", config.name, len(client._tools))
            except Exception as e:
                logger.error("Failed to connect to MCP server %s: %s", config.name, e)

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for name, client in self.clients.items():
            try:
                await client.disconnect()
            except Exception as e:
                logger.error("Error disconnecting MCP server %s: %s", name, e)
        self.clients.clear()

    def get_tools(self) -> list[McpTool]:
        """Get all tools from all connected servers."""
        tools: list[McpTool] = []
        for client in self.clients.values():
            tools.extend(client._tools)
        return tools

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on a specific server.

        Args:
            server_name: Name of the MCP server.
            tool_name: Name of the tool.
            arguments: Tool arguments.

        Returns:
            Tool result.
        """
        client = self.clients.get(server_name)
        if not client:
            raise KeyError(f"MCP server '{server_name}' not connected")

        return await client.call_tool(tool_name, arguments)


# ---------------------------------------------------------------------------
# Public helper — run MCP operations in a synchronous context
# ---------------------------------------------------------------------------


def _run_async(coro):
    """Run an async coroutine in a new event loop (for sync tool calls)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
