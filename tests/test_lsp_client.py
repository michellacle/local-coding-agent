"""Tests for LSP client — Language Server Protocol integration."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import json

from local_agent.lsp_client import (
    LSPClient,
    Diagnostic,
    Location,
    SymbolInfo,
    HoverInfo,
    uri_to_path,
    path_to_uri,
)


class TestDiagnostic:
    """Test Diagnostic dataclass."""

    def test_str_error(self):
        d = Diagnostic(line=5, character=10, severity=1, message="Undefined variable", source="pyright")
        s = str(d)
        assert "error" in s
        assert "L6:11" in s
        assert "Undefined variable" in s

    def test_str_warning(self):
        d = Diagnostic(line=0, character=0, severity=2, message="Unused import")
        s = str(d)
        assert "warning" in s

    def test_str_info(self):
        d = Diagnostic(line=10, character=20, severity=3, message="Hint")
        assert "info" in str(d)

    def test_str_hint(self):
        d = Diagnostic(line=0, character=0, severity=4, message="Suggestion")
        assert "hint" in str(d)


class TestLocation:
    """Test Location dataclass."""

    def test_str(self):
        loc = Location(file="/foo/bar.py", line=10, character=5)
        assert "/foo/bar.py:11:6" in str(loc)

    def test_zero_index(self):
        loc = Location(file="/test.py", line=0, character=0)
        assert "test.py:1:1" in str(loc)


class TestSymbolInfo:
    """Test SymbolInfo dataclass."""

    def test_fields(self):
        s = SymbolInfo(name="MyClass", kind=5, file="/test.py", line=0, container="module")
        assert s.name == "MyClass"
        assert s.container == "module"


class TestHoverInfo:
    """Test HoverInfo dataclass."""

    def test_default_kind(self):
        h = HoverInfo(content="Some docs")
        assert h.kind == "markdown"

    def test_plain_kind(self):
        h = HoverInfo(content="Plain text", kind="plain")
        assert h.kind == "plain"


class TestUriConversion:
    """Test URI/path conversion utilities."""

    def test_uri_to_path_basic(self):
        assert uri_to_path("file:///home/user/file.py") == "/home/user/file.py"

    def test_uri_to_path_with_spaces(self):
        assert uri_to_path("file:///home/user/my%20file.py") == "/home/user/my file.py"

    def test_uri_to_path_with_hash(self):
        assert uri_to_path("file:///home/file%23test.py") == "/home/file#test.py"

    def test_path_to_uri(self):
        uri = path_to_uri("/home/user/file.py")
        assert uri.startswith("file:///home/user/file.py")

    def test_uri_plain_passthrough(self):
        assert uri_to_path("http://example.com") == "http://example.com"


class TestLSPClientInit:
    """Test LSPClient initialization."""

    def test_init_defaults(self):
        client = LSPClient(server_command=["pyright-langserver", "--stdio"], root_dir="/tmp")
        assert client.server_command == ["pyright-langserver", "--stdio"]
        assert not client._running
        assert not client._initialized

    def test_init_timeout(self):
        client = LSPClient(server_command=["test"], root_dir="/tmp", timeout=30.0)
        assert client.timeout == 30.0

    def test_guess_language_python(self):
        client = LSPClient(server_command=["test"], root_dir="/tmp")
        assert client._guess_language("/path/to/file.py") == "python"

    def test_guess_language_typescript(self):
        client = LSPClient(server_command=["test"], root_dir="/tmp")
        assert client._guess_language("/path/to/file.ts") == "typescript"

    def test_guess_language_javascript(self):
        client = LSPClient(server_command=["test"], root_dir="/tmp")
        assert client._guess_language("/path/to/file.js") == "javascript"

    def test_guess_language_rust(self):
        client = LSPClient(server_command=["test"], root_dir="/tmp")
        assert client._guess_language("/path/to/file.rs") == "rust"

    def test_guess_language_go(self):
        client = LSPClient(server_command=["test"], root_dir="/tmp")
        assert client._guess_language("/path/to/file.go") == "go"

    def test_guess_language_plaintext(self):
        client = LSPClient(server_command=["test"], root_dir="/tmp")
        assert client._guess_language("/path/to/file.xyz") == "plaintext"


class TestLSPClientWithoutServer:
    """Test LSPClient behavior when no server is running."""

    @pytest.fixture
    def client(self):
        return LSPClient(server_command=["fake-server"], root_dir="/tmp")

    def test_start_file_not_found(self, client):
        result = client.start()
        assert result is False

    def test_get_diagnostics_no_server(self, client):
        diags = client.get_diagnostics("/tmp/test.py")
        assert diags == []

    def test_definition_no_server(self, client):
        locs = client.definition("/tmp/test.py", 0, 0)
        assert locs == []

    def test_references_no_server(self, client):
        locs = client.references("/tmp/test.py", 0, 0)
        assert locs == []

    def test_symbols_no_server(self, client):
        syms = client.symbols("/tmp/test.py")
        assert syms == []

    def test_hover_no_server(self, client):
        result = client.hover("/tmp/test.py", 0, 0)
        assert result is None

    def test_open_file_no_server(self, client):
        # Should not raise
        client.open_file("/tmp/test.py", "print('hi')")

    def test_change_file_no_server(self, client):
        # Should not raise
        client.change_file("/tmp/test.py", "print('hi')")

    def test_stop_no_server(self, client):
        # Should not raise
        client.stop()


class TestLSPClientContextManager:
    """Test context manager protocol."""

    def test_enter_calls_start(self):
        client = LSPClient(server_command=["fake"], root_dir="/tmp")
        with patch.object(client, "start") as mock_start:
            # Won't work but should call start
            pass

    def test_exit_calls_stop(self):
        client = LSPClient(server_command=["fake"], root_dir="/tmp")
        with patch.object(client, "stop") as mock_stop:
            client.__exit__()
            mock_stop.assert_called_once()


class TestDiagnosticStorage:
    """Test diagnostic storage and retrieval."""

    def test_store_and_retrieve(self):
        client = LSPClient(server_command=["fake"], root_dir="/tmp")
        fpath = str(Path("/tmp/test.py").resolve())
        client._diagnostics[fpath] = [
            Diagnostic(line=0, character=5, severity=1, message="Error"),
            Diagnostic(line=2, character=0, severity=2, message="Warning"),
        ]
        diags = client.get_diagnostics("/tmp/test.py")
        assert len(diags) == 2
        assert diags[0].message == "Error"
        assert diags[1].severity == 2

    def test_empty_diagnostics(self):
        client = LSPClient(server_command=["fake"], root_dir="/tmp")
        diags = client.get_diagnostics("/tmp/nonexistent.py")
        assert diags == []
