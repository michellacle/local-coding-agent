"""Tests for browser_engine — BrowserEngine with mocked Playwright."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from local_agent.browser_engine import BrowserEngine, ConsoleMessage


class TestBrowserEngineNoPlaywright:
    """Test behavior when Playwright is not installed."""

    def test_navigate_raises_without_playwright(self):
        engine = BrowserEngine()
        with pytest.raises(RuntimeError, match="Playwright is not installed"):
            engine.navigate("http://example.com")


class TestBrowserEngineConsole:
    """Test console message capture (no Playwright needed)."""

    def test_console_capture(self):
        engine = BrowserEngine()

        mock_msg = MagicMock()
        mock_msg.type = "log"
        mock_msg.text = "hello world"

        engine._handle_console(mock_msg)

        output = engine.get_console_output()
        assert len(output) == 1
        assert output[0]["type"] == "log"
        assert output[0]["text"] == "hello world"

    def test_clear_console(self):
        engine = BrowserEngine()

        mock_msg = MagicMock()
        mock_msg.type = "error"
        mock_msg.text = "oops"
        engine._handle_console(mock_msg)

        engine.clear_console()
        assert len(engine.get_console_output()) == 0

    def test_console_message_dataclass(self):
        msg = ConsoleMessage(type="warn", text="deprecated", timestamp=1234567890.0)
        assert msg.type == "warn"
        assert msg.text == "deprecated"
        assert msg.timestamp == 1234567890.0


class TestBrowserEngineConfig:
    """Test browser engine configuration."""

    def test_default_headless(self):
        engine = BrowserEngine()
        assert engine._headless is True

    def test_custom_viewport(self):
        engine = BrowserEngine(viewport_width=1920, viewport_height=1080)
        assert engine._viewport_width == 1920
        assert engine._viewport_height == 1080

    def test_custom_timeout(self):
        engine = BrowserEngine(timeout=60000)
        assert engine._timeout == 60000

    def test_close(self):
        engine = BrowserEngine()
        engine.close()
        assert engine._closed is True


class TestBrowserEngineWithMockedPlaywright:
    """Test browser engine with fully mocked Playwright (inject module into sys.modules)."""

    def _inject_mock_playwright(self):
        """Inject a fake playwright module into sys.modules."""
        mock_page = MagicMock()
        mock_page.title.return_value = "Test Page"
        mock_page.url = "http://test.example.com"

        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page

        mock_browser = MagicMock()
        mock_browser.new_context.return_value = mock_context
        mock_browser.new_context.return_value.new_page.return_value = mock_page
        mock_browser.close.return_value = None

        mock_sync_api = MagicMock()
        mock_sync_api.sync_playwright = MagicMock()

        # Make sync_playwright a callable context manager
        mock_sp_instance = MagicMock()
        mock_sp_instance.chromium = MagicMock()
        mock_sp_instance.chromium.launch.return_value = mock_browser
        mock_sp_instance.__enter__ = MagicMock(return_value=mock_sp_instance)
        mock_sp_instance.__exit__ = MagicMock(return_value=False)
        mock_sync_api.sync_playwright.return_value = mock_sp_instance

        return mock_sync_api, mock_page

    def test_navigate_success(self):
        engine = BrowserEngine()
        mock_sync_api, mock_page = self._inject_mock_playwright()

        with patch.dict("sys.modules", {"playwright": MagicMock(), "playwright.sync_api": mock_sync_api}):
            result = engine.navigate("http://test.example.com")
            assert result["status"] == "loaded"

    def test_navigate_timeout_error(self):
        engine = BrowserEngine()

        mock_page = MagicMock()
        mock_page.goto = MagicMock(side_effect=Exception("net::ERR_TIMEOUT"))

        mock_browser = MagicMock()
        mock_browser.new_context.return_value.new_page.return_value = mock_page
        mock_browser.close.return_value = None

        mock_sp_instance = MagicMock()
        mock_sp_instance.chromium = MagicMock()
        mock_sp_instance.chromium.launch.return_value = mock_browser
        mock_sp_instance.__enter__ = MagicMock(return_value=mock_sp_instance)
        mock_sp_instance.__exit__ = MagicMock(return_value=False)

        mock_sync_api = MagicMock()
        mock_sync_api.sync_playwright.return_value = mock_sp_instance

        with patch.dict("sys.modules", {"playwright": MagicMock(), "playwright.sync_api": mock_sync_api}):
            result = engine.navigate("http://slow.example.com")
            assert "error" in result["status"]
