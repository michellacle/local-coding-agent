"""LSP Client — Language Server Protocol integration for IDE-like code intelligence.

Provides diagnostics, go-to-definition, symbol search, hover info, and
references — connecting to language servers like pyright, typescript-language-server,
rust-analyzer, etc.

Uses the JSON-RPC stdio transport (same as most LSP servers).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Diagnostic:
    """A single diagnostic (error/warning/info) from an LSP server."""

    line: int          # 0-based
    character: int     # 0-based column
    severity: int      # 1=error, 2=warning, 3=info, 4=hint
    message: str
    source: str = ""
    code: str = ""

    def __str__(self) -> str:
        sev_map = {1: "error", 2: "warning", 3: "info", 4: "hint"}
        sev = sev_map.get(self.severity, f"s{self.severity}")
        return f"[{sev}] L{self.line + 1}:{self.character + 1} {self.message}"


@dataclass
class Location:
    """A location in a file (for go-to-definition, references, etc.)."""

    file: str
    line: int          # 0-based
    character: int     # 0-based

    def __str__(self) -> str:
        return f"{self.file}:{self.line + 1}:{self.character + 1}"


@dataclass
class SymbolInfo:
    """A symbol found in a file or workspace."""

    name: str
    kind: int          # LSP SymbolKind
    file: str
    line: int          # 0-based
    container: str = ""


@dataclass
class HoverInfo:
    """Hover information for a symbol."""

    content: str
    kind: str = "markdown"


class LSPClient:
    """Lightweight LSP client using JSON-RPC over stdio.

    Connects to a language server process, initializes it, and provides
    convenience methods for common LSP operations.

    Usage:
        lsp = LSPClient(server_command=["pyright-langserver", "--stdio"], root_dir="/path/to/project")
        lsp.start()
        diagnostics = lsp.get_diagnostics("/path/to/project/foo.py")
        definitions = lsp.definition("/path/to/project/foo.py", line=10, character=5)
        lsp.stop()
    """

    def __init__(
        self,
        server_command: list[str],
        root_dir: str,
        timeout: float = 10.0,
    ) -> None:
        """Initialize the LSP client.

        Args:
            server_command: Command to start the language server.
            root_dir: Root directory of the project.
            timeout: Timeout for JSON-RPC requests (seconds).
        """
        self.server_command = server_command
        self.root_dir = Path(root_dir).resolve()
        self.timeout = timeout
        self._process: subprocess.Popen[bytes] | None = None
        self._request_id = 0
        self._lock = threading.Lock()
        self._running = False
        self._initialized = False
        self._diagnostics: dict[str, list[Diagnostic]] = {}

    def start(self) -> bool:
        """Start the language server process and initialize it.

        Returns:
            True if the server started and initialized successfully.
        """
        try:
            self._process = subprocess.Popen(
                self.server_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.root_dir),
            )
            self._running = True

            # Wait a moment for the server to be ready
            time.sleep(0.5)

            # Initialize
            result = self._send_request(
                "initialize",
                {
                    "processId": os.getpid(),
                    "rootUri": f"file://{self.root_dir}",
                    "capabilities": {
                        "textDocument": {
                            "diagnostic": {"dynamicRegistration": False},
                            "publishDiagnostics": {"relatedInformation": True},
                        },
                        "window": {
                            "workDoneProgress": False,
                        },
                    },
                },
                timeout=self.timeout,
            )

            if result is None:
                logger.warning("LSP server returned no initialize result")
                return False

            self._initialized = True

            # Send initialized notification
            self._send_notification("initialized", {})

            # Register for diagnostics
            self._register_diagnostics()

            return True

        except FileNotFoundError:
            logger.warning("LSP server not found: %s", self.server_command[0])
            return False
        except Exception as e:
            logger.warning("Failed to start LSP server: %s", e)
            return False

    def stop(self) -> None:
        """Shutdown and stop the language server."""
        if self._process and self._running:
            try:
                self._send_notification("shutdown", {})
                self._send_notification("exit", {})
            except Exception:
                pass
            self._process.stdin.close() if self._process.stdin else None
            self._process.wait(timeout=5)
            self._running = False
            self._initialized = False

    def get_diagnostics(self, file_path: str | Path) -> list[Diagnostic]:
        """Get diagnostics for a file.

        Args:
            file_path: Path to the file.

        Returns:
            List of diagnostics for the file.
        """
        fpath = str(Path(file_path).resolve())
        return self._diagnostics.get(fpath, [])

    def definition(self, file_path: str, line: int, character: int) -> list[Location]:
        """Go to definition at a position.

        Args:
            file_path: Path to the file.
            line: 0-based line number.
            character: 0-based character offset.

        Returns:
            List of locations where the definition is found.
        """
        if not self._initialized:
            return []

        positions = self._request_definition(file_path, line, character)
        locations = []
        for pos in positions:
            uri = pos.get("uri", "")
            rng = pos.get("range", {})
            loc = Location(
                file=uri_to_path(uri),
                line=rng.get("start", {}).get("line", 0),
                character=rng.get("start", {}).get("character", 0),
            )
            locations.append(loc)

        # Also handle DefinitionLink format
        if not locations and positions:
            for link in positions:
                if "targetUri" in link:
                    rng = link.get("targetSelectionRange", {})
                    locations.append(Location(
                        file=uri_to_path(link["targetUri"]),
                        line=rng.get("start", {}).get("line", 0),
                        character=rng.get("start", {}).get("character", 0),
                    ))

        return locations

    def references(self, file_path: str, line: int, character: int, include_declaration: bool = False) -> list[Location]:
        """Find all references to a symbol.

        Args:
            file_path: Path to the file.
            line: 0-based line number.
            character: 0-based character offset.
            include_declaration: Whether to include the declaration itself.

        Returns:
            List of locations where the symbol is referenced.
        """
        if not self._initialized:
            return []

        positions = self._request_references(file_path, line, character, include_declaration)
        return [
            Location(
                file=uri_to_path(p["uri"]),
                line=p["range"]["start"]["line"],
                character=p["range"]["start"]["character"],
            )
            for p in positions
        ]

    def symbols(self, file_path: str) -> list[SymbolInfo]:
        """List symbols in a file.

        Args:
            file_path: Path to the file.

        Returns:
            List of symbols found in the file.
        """
        if not self._initialized:
            return []

        result = self._send_request(
            "textDocument/documentSymbol",
            {
                "textDocument": {"uri": path_to_uri(file_path)},
            },
        )

        symbols = []
        if result:
            self._extract_symbols(result, str(file_path), "")
            self._flatten_symbols(result, str(file_path), symbols, "")

        return symbols

    def hover(self, file_path: str, line: int, character: int) -> HoverInfo | None:
        """Get hover information for a symbol at a position.

        Args:
            file_path: Path to the file.
            line: 0-based line number.
            character: 0-based character offset.

        Returns:
            HoverInfo if available, None otherwise.
        """
        if not self._initialized:
            return None

        result = self._send_request(
            "textDocument/hover",
            {
                "textDocument": {"uri": path_to_uri(file_path)},
                "position": {"line": line, "character": character},
            },
        )

        if not result or not result.get("contents"):
            return None

        contents = result["contents"]
        if isinstance(contents, str):
            return HoverInfo(content=contents, kind="plain")
        elif isinstance(contents, dict):
            kind = contents.get("kind", "markdown")
            value = contents.get("value", "")
            return HoverInfo(content=value, kind=kind)
        elif isinstance(contents, list):
            return HoverInfo(content="\n".join(str(c) for c in contents), kind="markdown")

        return None

    def open_file(self, file_path: str, content: str) -> None:
        """Notify the server that a file has been opened/changed.

        This triggers diagnostics for the file.

        Args:
            file_path: Path to the file.
            content: Current file content.
        """
        if not self._initialized:
            return

        self._send_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": path_to_uri(file_path),
                "languageId": self._guess_language(file_path),
                "version": 1,
                "text": content,
            },
        })

    def change_file(self, file_path: str, content: str, version: int = 1) -> None:
        """Notify the server that a file content has changed.

        Args:
            file_path: Path to the file.
            content: New file content.
            version: Document version number.
        """
        if not self._initialized:
            return

        self._send_notification("textDocument/didChange", {
            "textDocument": {
                "uri": path_to_uri(file_path),
                "version": version,
            },
            "contentChanges": [{"text": content}],
        })

    # ------------------------------------------------------------------ #
    #  Internal methods                                                     #
    # ------------------------------------------------------------------ #

    def _request_definition(self, file_path: str, line: int, character: int) -> list[Any]:
        result = self._send_request(
            "textDocument/definition",
            {
                "textDocument": {"uri": path_to_uri(file_path)},
                "position": {"line": line, "character": character},
            },
        )
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "targets" in result:
            return result["targets"]
        if isinstance(result, dict):
            return [result]
        return []

    def _request_references(
        self, file_path: str, line: int, character: int, include_declaration: bool
    ) -> list[Any]:
        result = self._send_request(
            "textDocument/references",
            {
                "textDocument": {"uri": path_to_uri(file_path)},
                "position": {"line": line, "character": character},
                "context": {"includeDeclaration": include_declaration},
            },
        )
        return result if isinstance(result, list) else []

    def _extract_symbols(self, items: list | Any, file_path: str, container: str) -> None:
        """Extract symbols recursively."""
        if isinstance(items, list):
            for item in items:
                self._extract_symbols(item, file_path, container)

    def _flatten_symbols(self, items: list | Any, file_path: str, result: list[SymbolInfo], container: str) -> None:
        """Flatten symbol tree into flat list."""
        if isinstance(items, list):
            for item in items:
                name = item.get("name", "")
                kind = item.get("kind", 0)
                rng = item.get("range", {})
                line = rng.get("start", {}).get("line", 0)
                children = item.get("children", [])
                new_container = name if children else container
                result.append(SymbolInfo(
                    name=name, kind=kind, file=file_path, line=line, container=container,
                ))
                if children:
                    self._flatten_symbols(children, file_path, result, new_container)
        elif isinstance(items, dict):
            pass  # Handle dict if needed

    def _register_diagnostics(self) -> None:
        """Register for diagnostic pull mode if available."""
        pass  # Most servers push diagnostics via notification

    def _guess_language(self, file_path: str) -> str:
        """Guess language ID from file extension."""
        ext_map = {
            ".py": "python",
            ".ts": "typescript",
            ".tsx": "typescriptreact",
            ".js": "javascript",
            ".jsx": "javascriptreact",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".rb": "ruby",
            ".rs": "rust",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".md": "markdown",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".sh": "shellscript",
            ".bash": "shellscript",
        }
        ext = Path(file_path).suffix.lower()
        return ext_map.get(ext, "plaintext")

    # ------------------------------------------------------------------ #
    #  JSON-RPC transport                                                   #
    # ------------------------------------------------------------------ #

    def _next_id(self) -> int:
        with self._lock:
            self._request_id += 1
            return self._request_id

    def _send_request(self, method: str, params: dict, timeout: float = 10.0) -> Any | None:
        """Send a JSON-RPC request and wait for the response."""
        if not self._process or not self._running:
            return None

        req_id = self._next_id()
        message = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        payload = f"Content-Length: {len(message_str := json.dumps(message))}\r\n\r\n{message_str}"

        try:
            self._process.stdin.write(payload.encode())  # type: ignore[union-attr]
            self._process.stdin.flush()  # type: ignore[union-attr]

            # Wait for response with matching ID
            deadline = time.time() + timeout
            while time.time() < deadline:
                resp = self._read_response()
                if resp and resp.get("id") == req_id:
                    if "result" in resp:
                        return resp["result"]
                    elif "error" in resp:
                        logger.debug("LSP error: %s", resp["error"])
                        return None
                    return None

            return None
        except Exception as e:
            logger.debug("LSP request failed: %s", e)
            return None

    def _send_notification(self, method: str, params: dict) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self._process or not self._running:
            return

        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        payload = f"Content-Length: {len(message_str := json.dumps(message))}\r\n\r\n{message_str}"

        try:
            self._process.stdin.write(payload.encode())  # type: ignore[union-attr]
            self._process.stdin.flush()  # type: ignore[union-attr]
        except Exception:
            pass

    def _read_response(self) -> dict | None:
        """Read a single JSON-RPC response from the server."""
        if not self._process or not self._running:
            return None

        # Read content-length header
        header = b""
        while True:
            byte = self._process.stdout.read(1)  # type: ignore[union-attr]
            if not byte:
                return None
            header += byte
            if b"\r\n\r\n" in header:
                break

        # Parse content-length
        header_str = header.decode("utf-8", errors="replace")
        content_length = 0
        for line in header_str.split("\r\n"):
            if line.lower().startswith("content-length:"):
                content_length = int(line.split(":")[1].strip())
                break

        if content_length == 0:
            return None

        # Read content
        content = b""
        while len(content) < content_length:
            chunk = self._process.stdout.read(content_length - len(content))  # type: ignore[union-attr]
            if not chunk:
                break
            content += chunk

        try:
            return json.loads(content.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def __enter__(self) -> "LSPClient":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()


def uri_to_path(uri: str) -> str:
    """Convert a file URI to a filesystem path."""
    if uri.startswith("file://"):
        path = uri[7:]
        # Handle percent-encoding
        path = path.replace("%20", " ").replace("%23", "#").replace("%3A", ":")
        # Windows UNC paths: file:///C:/... -> C:/...
        if len(path) > 2 and path.startswith("/") and path[1].isalpha() and len(path) > 2 and path[2] == ":":
            path = path[1:]
        return path
    return uri


def path_to_uri(file_path: str) -> str:
    """Convert a filesystem path to a file URI."""
    p = Path(file_path).resolve()
    return f"file://{p}"
