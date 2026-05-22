"""Tests for browser engine — Playwright-based browser automation."""

from unittest.mock import MagicMock, patch

import pytest

from local_agent.browser_engine import BrowserEngine, ConsoleMessage


class TestConsoleMessage:
    """Test ConsoleMessage dataclass."""

    def test_create(self):
        msg = ConsoleMessage(type="log", text="hello", timestamp=123.0)
        assert msg.type == "log"
        assert msg.text == "hello"
        assert msg.timestamp == 123.0


class TestBrowserEngineNoPlaywright:
    """Test BrowserEngine behavior when Playwright is not installed."""

    def test_navigate_raises_without_playwright(self):
        engine = BrowserEngine()
        with patch.dict("sys.modules", {"playwright": None, "playwright.sync_api": None}):
            with pytest.raises(RuntimeError, match="Playwright is not installed"):
                engine.navigate("https://example.com")

    def test_click_raises_without_playwright(self):
        engine = BrowserEngine()
        with patch.dict("sys.modules", {"playwright": None, "playwright.sync_api": None}):
            with pytest.raises(RuntimeError, match="Playwright is not installed"):
                engine.click("#submit")

    def test_type_text_raises_without_playwright(self):
        engine = BrowserEngine()
        with patch.dict("sys.modules", {"playwright": None, "playwright.sync_api": None}):
            with pytest.raises(RuntimeError, match="Playwright is not installed"):
                engine.type_text("#input", "hello")

    def test_screenshot_raises_without_playwright(self):
        engine = BrowserEngine()
        with patch.dict("sys.modules", {"playwright": None, "playwright.sync_api": None}):
            with pytest.raises(RuntimeError, match="Playwright is not installed"):
                engine.screenshot("/tmp/snap.png")


class TestBrowserEngineInit:
    """Test BrowserEngine initialization."""

    def test_default_init(self):
        engine = BrowserEngine()
        assert engine._headless is True
        assert engine._viewport_width == 1280
        assert engine._viewport_height == 720
        assert engine._timeout == 30000
        assert engine._closed is True

    def test_custom_init(self):
        engine = BrowserEngine(
            headless=False,
            viewport_width=1920,
            viewport_height=1080,
            timeout=60000,
        )
        assert engine._headless is False
        assert engine._viewport_width == 1920
        assert engine._viewport_height == 1080
        assert engine._timeout == 60000


class TestBrowserEngineConsole:
    """Test console message handling."""

    def test_get_console_output_empty(self):
        engine = BrowserEngine()
        output = engine.get_console_output()
        assert output == []

    def test_clear_console(self):
        engine = BrowserEngine()
        engine._console_messages = [
            ConsoleMessage(type="log", text="test", timestamp=1.0),
        ]
        engine.clear_console()
        assert engine._console_messages == []

    def test_handle_console(self):
        engine = BrowserEngine()
        mock_msg = MagicMock()
        mock_msg.type = "error"
        mock_msg.text = "Uncaught TypeError"
        engine._handle_console(mock_msg)
        assert len(engine._console_messages) == 1
        assert engine._console_messages[0].type == "error"
        assert engine._console_messages[0].text == "Uncaught TypeError"

    def test_console_output_format(self):
        engine = BrowserEngine()
        engine._console_messages = [
            ConsoleMessage(type="warn", text="deprecated", timestamp=100.0),
        ]
        output = engine.get_console_output()
        assert len(output) == 1
        assert output[0]["type"] == "warn"
        assert output[0]["text"] == "deprecated"


class TestBrowserEngineExtractAccessibility:
    """Test accessibility tree extraction."""

    def test_extract_with_snapshot(self):
        engine = BrowserEngine()
        mock_page = MagicMock()
        mock_page.accessibility.snapshot.return_value = {
            "role": "root",
            "children": [{"role": "heading", "name": "Hello"}],
        }
        result = engine._extract_accessibility_tree(mock_page)
        assert "root" in result
        assert "Hello" in result

    def test_extract_fallback(self):
        engine = BrowserEngine()
        mock_page = MagicMock()
        mock_page.accessibility.snapshot.side_effect = Exception("no a11y")
        mock_page.inner_text.return_value = "Page text content"
        result = engine._extract_accessibility_tree(mock_page)
        assert result == "Page text content"


class TestBrowserEngineMocked:
    """Test BrowserEngine methods with mocked Playwright."""

    def _mock_sync_playwright(self):
        """Create a mock sync_playwright context manager."""
        mock_p = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_p.__enter__ = MagicMock(return_value=mock_p)
        mock_p.__exit__ = MagicMock(return_value=False)
        mock_p.chromium.launch.return_value = mock_browser
        mock_browser.new_context = MagicMock(return_value=mock_context)
        mock_context.new_page = MagicMock(return_value=mock_page)

        return mock_p, mock_browser, mock_context, mock_page

    def _inject_mock_playwright(self, mock_sp):
        """Inject a fake playwright.sync_api module into sys.modules."""
        import sys
        mock_module = MagicMock()
        mock_module.sync_playwright = mock_sp
        sys.modules["playwright"] = MagicMock()
        sys.modules["playwright.sync_api"] = mock_module
        return mock_module

    def _cleanup_playwright(self):
        """Remove fake playwright modules from sys.modules."""
        import sys
        for key in list(sys.modules.keys()):
            if key.startswith("playwright"):
                del sys.modules[key]

    def test_navigate_success(self):
        mock_sp = MagicMock()
        mock_p, mock_browser, mock_context, mock_page = self._mock_sync_playwright()
        mock_sp.return_value = mock_p
        mock_page.goto = MagicMock()
        mock_page.url = "https://example.com"
        mock_page.title.return_value = "Example"
        self._inject_mock_playwright(mock_sp)

        engine = BrowserEngine()
        result = engine.navigate("https://example.com")

        self._cleanup_playwright()
        assert result["status"] == "loaded"
        assert result["url"] == "https://example.com"
        assert result["title"] == "Example"
        mock_browser.close.assert_called_once()

    def test_navigate_error(self):
        mock_sp = MagicMock()
        mock_p, mock_browser, mock_context, mock_page = self._mock_sync_playwright()
        mock_sp.return_value = mock_p
        mock_page.goto.side_effect = Exception("timeout")
        self._inject_mock_playwright(mock_sp)

        engine = BrowserEngine()
        result = engine.navigate("https://example.com")

        self._cleanup_playwright()
        assert result["status"].startswith("error:")
        assert "timeout" in result["status"]

    def test_click_success(self):
        mock_sp = MagicMock()
        mock_p, mock_browser, mock_context, mock_page = self._mock_sync_playwright()
        mock_sp.return_value = mock_p
        mock_page.click.return_value = True
        self._inject_mock_playwright(mock_sp)

        engine = BrowserEngine()
        result = engine.click("#button")

        self._cleanup_playwright()
        assert result["status"] == "success"

    def test_click_error(self):
        mock_sp = MagicMock()
        mock_p, mock_browser, mock_context, mock_page = self._mock_sync_playwright()
        mock_sp.return_value = mock_p
        mock_page.click.side_effect = Exception("not found")
        self._inject_mock_playwright(mock_sp)

        engine = BrowserEngine()
        result = engine.click("#missing")

        self._cleanup_playwright()
        assert result["status"] == "error"
        assert "not found" in result["error"]

    def test_type_text(self):
        mock_sp = MagicMock()
        mock_p, mock_browser, mock_context, mock_page = self._mock_sync_playwright()
        mock_sp.return_value = mock_p
        mock_page.fill.return_value = True
        self._inject_mock_playwright(mock_sp)

        engine = BrowserEngine()
        result = engine.type_text("#input", "hello world")

        self._cleanup_playwright()
        assert result["status"] == "success"
        mock_page.fill.assert_called_once_with("#input", "hello world")

    def test_press_key(self):
        mock_sp = MagicMock()
        mock_p, mock_browser, mock_context, mock_page = self._mock_sync_playwright()
        mock_sp.return_value = mock_p
        mock_page.keyboard.press.return_value = True
        self._inject_mock_playwright(mock_sp)

        engine = BrowserEngine()
        result = engine.press_key("Enter")

        self._cleanup_playwright()
        assert result["status"] == "success"
        mock_page.keyboard.press.assert_called_once_with("Enter")

    def test_screenshot(self):
        mock_sp = MagicMock()
        mock_p, mock_browser, mock_context, mock_page = self._mock_sync_playwright()
        mock_sp.return_value = mock_p
        mock_page.screenshot.return_value = b"fake_png_data"
        self._inject_mock_playwright(mock_sp)

        engine = BrowserEngine()
        result = engine.screenshot("/tmp/test.png")

        self._cleanup_playwright()
        assert result["status"] == "success"

    def test_get_page_content(self):
        mock_sp = MagicMock()
        mock_p, mock_browser, mock_context, mock_page = self._mock_sync_playwright()
        mock_sp.return_value = mock_p
        mock_page.content.return_value = "<html><body>hi</body></html>"
        self._inject_mock_playwright(mock_sp)

        engine = BrowserEngine()
        result = engine.get_page_content()

        self._cleanup_playwright()
        assert result["status"] == "success"

    def test_evaluate(self):
        mock_sp = MagicMock()
        mock_p, mock_browser, mock_context, mock_page = self._mock_sync_playwright()
        mock_sp.return_value = mock_p
        mock_page.evaluate.return_value = 42
        self._inject_mock_playwright(mock_sp)

        engine = BrowserEngine()
        result = engine.evaluate("2 + 2")

        self._cleanup_playwright()
        assert result["status"] == "success"

    def test_accessibility_tree(self):
        mock_sp = MagicMock()
        mock_p, mock_browser, mock_context, mock_page = self._mock_sync_playwright()
        mock_sp.return_value = mock_p
        mock_page.accessibility.snapshot.return_value = {"role": "root"}
        self._inject_mock_playwright(mock_sp)

        engine = BrowserEngine()
        result = engine.get_accessibility_tree()

        self._cleanup_playwright()
        assert result["status"] == "success"

    def test_close(self):
        engine = BrowserEngine()
        engine.close()
        assert engine._closed is True

    def test_viewport_config_applied(self):
        mock_sp = MagicMock()
        mock_p, mock_browser, mock_context, mock_page = self._mock_sync_playwright()
        mock_sp.return_value = mock_p
        mock_page.goto = MagicMock()
        mock_page.url = "https://example.com"
        mock_page.title.return_value = "Test"
        self._inject_mock_playwright(mock_sp)

        engine = BrowserEngine(viewport_width=1920, viewport_height=1080)
        engine.navigate("https://example.com")

        self._cleanup_playwright()
        mock_browser.new_context.assert_called_with(
            viewport={"width": 1920, "height": 1080}
        )
