"""Tests for MCP client — transport layer, JSON-RPC, and tool discovery."""

import asyncio
import json
import pytest
import subprocess
import tempfile
import os

from local_agent.mcp_client import (
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcError,
    McpTransportConfig,
    StdioTransport,
    McpClient,
    McpManager,
    McpTool,
    _run_async,
)


class TestJsonRpcRequest:
    def test_to_dict(self):
        req = JsonRpcRequest(id=1, method="tools/list", params={"key": "val"})
        d = req.to_dict()
        assert d == {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {"key": "val"},
        }

    def test_to_dict_no_params(self):
        req = JsonRpcRequest(id=1, method="tools/list")
        d = req.to_dict()
        assert "params" not in d

    def test_to_json(self):
        req = JsonRpcRequest(id=42, method="initialize", params={"a": 1})
        raw = req.to_json()
        parsed = json.loads(raw)
        assert parsed["id"] == 42
        assert parsed["method"] == "initialize"


class TestJsonRpcResponse:
    def test_from_dict_success(self):
        d = {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}
        resp = JsonRpcResponse.from_dict(d)
        assert resp.id == 1
        assert resp.result == {"tools": []}
        assert resp.error is None

    def test_from_dict_error(self):
        d = {
            "jsonrpc": "2.0",
            "id": 2,
            "error": {"code": -32600, "message": "Invalid Request"},
        }
        resp = JsonRpcResponse.from_dict(d)
        assert resp.error is not None

    def test_raise_for_error_success(self):
        resp = JsonRpcResponse(id=1, result="ok")
        result = resp.raise_for_error()
        assert result == "ok"

    def test_raise_for_error_raises(self):
        resp = JsonRpcResponse(
            id=1,
            error={"code": -32601, "message": "Method not found"},
        )
        with pytest.raises(JsonRpcError) as exc_info:
            resp.raise_for_error()
        assert exc_info.value.code == -32601
        assert "Method not found" in str(exc_info.value)


class TestMcpTool:
    def test_creation(self):
        tool = McpTool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
            server_name="test-server",
        )
        assert tool.name == "test_tool"
        assert tool.server_name == "test-server"


class TestStdioTransport:
    """Test stdio transport with a mock MCP server script."""

    @pytest.fixture
    def mock_server_script(self):
        """Create a mock MCP server script that responds to JSON-RPC."""
        script = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="mock_mcp_"
        )
        script.write(
            '''#!/usr/bin/env python3
"""Mock MCP server for testing."""
import sys
import json

def handle_request(req):
    method = req.get("method", "")
    rid = req.get("id", 0)

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": rid,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "mock-server", "version": "0.1.0"},
                "capabilities": {"tools": {}},
            }
        }
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0", "id": rid,
            "result": {
                "tools": [
                    {
                        "name": "mock_tool",
                        "description": "A mock tool for testing",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "arg1": {"type": "string", "description": "First arg"}
                            }
                        }
                    }
                ]
            }
        }
    elif method == "tools/call":
        name = req.get("params", {}).get("name", "")
        args = req.get("params", {}).get("arguments", {})
        return {
            "jsonrpc": "2.0", "id": rid,
            "result": {
                "content": [{"type": "text", "text": f"Called {name} with {args}"}]
            }
        }
    elif method.startswith("notifications/"):
        return None  # no response for notifications
    else:
        return {
            "jsonrpc": "2.0", "id": rid,
            "error": {"code": -32601, "message": f"Unknown method: {method}"}
        }

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
            if resp is not None:
                print(json.dumps(resp), flush=True)
        except json.JSONDecodeError:
            pass

if __name__ == "__main__":
    main()
'''
        )
        script.flush()
        script.close()
        os.chmod(script.name, 0o755)
        yield script.name
        os.unlink(script.name)

    def test_connect_and_disconnect(self, mock_server_script):
        config = McpTransportConfig(
            name="mock-stdio",
            transport="stdio",
            command="python3",
            args=[mock_server_script],
        )
        transport = StdioTransport(config)

        async def run():
            await transport.connect()
            await transport.disconnect()

        asyncio.run(run())

    def test_send_request(self, mock_server_script):
        config = McpTransportConfig(
            name="mock-stdio",
            transport="stdio",
            command="python3",
            args=[mock_server_script],
        )
        transport = StdioTransport(config)

        async def run():
            await transport.connect()
            req = JsonRpcRequest(id=1, method="tools/list")
            resp = await transport.send(req)
            assert resp.error is None
            assert "tools" in resp.result
            assert len(resp.result["tools"]) == 1
            await transport.disconnect()

        asyncio.run(run())

    def test_send_notification(self, mock_server_script):
        config = McpTransportConfig(
            name="mock-stdio",
            transport="stdio",
            command="python3",
            args=[mock_server_script],
        )
        transport = StdioTransport(config)

        async def run():
            await transport.connect()
            await transport.send_notification("notifications/initialized")
            # Should not raise
            await transport.disconnect()

        asyncio.run(run())


class TestMcpClient:
    @pytest.fixture
    def mock_server_script(self):
        script = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="mock_mcp_"
        )
        script.write(
            '''#!/usr/bin/env python3
import sys, json
def handle_request(req):
    method = req.get("method", "")
    rid = req.get("id", 0)
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "mock-server", "version": "0.1.0"},
            "capabilities": {"tools": {}},
        }}
    elif method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "tools": [
                {"name": "mock_tool", "description": "A mock tool",
                 "inputSchema": {"type": "object", "properties": {
                     "arg1": {"type": "string", "description": "First arg"}
                 }}}
            ]
        }}
    elif method == "tools/call":
        name = req.get("params", {}).get("name", "")
        args = req.get("params", {}).get("arguments", {})
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "content": [{"type": "text", "text": f"Called {name} with {args}"}]
        }}
    elif method.startswith("notifications/"):
        return None
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"Unknown: {method}"}}

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        resp = handle_request(req)
        if resp is not None:
            print(json.dumps(resp), flush=True)
    except json.JSONDecodeError:
        pass
'''
        )
        script.flush()
        script.close()
        os.chmod(script.name, 0o755)
        yield script.name
        os.unlink(script.name)

    def test_connect_and_list_tools(self, mock_server_script):
        config = McpTransportConfig(
            name="mock",
            transport="stdio",
            command="python3",
            args=[mock_server_script],
        )
        client = McpClient(config)

        async def run():
            await client.connect()
            assert client.is_connected
            tools = await client.list_tools()
            assert len(tools) == 1
            assert tools[0].name == "mock_tool"
            assert tools[0].server_name == "mock"
            await client.disconnect()

        asyncio.run(run())

    def test_call_tool(self, mock_server_script):
        config = McpTransportConfig(
            name="mock",
            transport="stdio",
            command="python3",
            args=[mock_server_script],
        )
        client = McpClient(config)

        async def run():
            await client.connect()
            result = await client.call_tool("mock_tool", {"arg1": "hello"})
            assert "Called mock_tool" in str(result)
            assert "hello" in str(result)
            await client.disconnect()

        asyncio.run(run())

    def test_connect_bad_transport(self):
        config = McpTransportConfig(
            name="bad",
            transport="websocket",
        )
        with pytest.raises(ValueError, match="Unknown transport"):
            McpClient(config)


class TestMcpManager:
    def test_get_tools_empty(self):
        manager = McpManager([])
        tools = manager.get_tools()
        assert tools == []

    @pytest.fixture
    def mock_server_script(self):
        script = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="mock_mcp_mgr_"
        )
        script.write(
            '''#!/usr/bin/env python3
import sys, json
def handle_request(req):
    method = req.get("method", "")
    rid = req.get("id", 0)
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "mock", "version": "0.1.0"},
            "capabilities": {"tools": {}},
        }}
    elif method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "tools": [
                {"name": "greet", "description": "Say hello",
                 "inputSchema": {"type": "object", "properties": {
                     "name": {"type": "string"}
                 }}}
            ]
        }}
    elif method == "tools/call":
        name = req.get("params", {}).get("name", "")
        args = req.get("params", {}).get("arguments", {})
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "content": [{"type": "text", "text": f"Hello, {args.get('name', 'world')}!"}]
        }}
    elif method.startswith("notifications/"):
        return None
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"Unknown: {method}"}}

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        resp = handle_request(req)
        if resp is not None:
            print(json.dumps(resp), flush=True)
    except json.JSONDecodeError:
        pass
'''
        )
        script.flush()
        script.close()
        os.chmod(script.name, 0o755)
        yield script.name
        os.unlink(script.name)

    def test_connect_all(self, mock_server_script):
        config = McpTransportConfig(
            name="test-server",
            transport="stdio",
            command="python3",
            args=[mock_server_script],
        )
        manager = McpManager([config])

        async def run():
            await manager.connect_all()
            tools = manager.get_tools()
            assert len(tools) == 1
            assert tools[0].name == "greet"
            await manager.call_tool("test-server", "greet", {"name": "Alice"})
            await manager.disconnect_all()

        asyncio.run(run())

    def test_call_tool_unknown_server(self):
        manager = McpManager([])
        with pytest.raises(KeyError, match="not connected"):
            _run_async(manager.call_tool("nope", "tool", {}))
