"""Browser Automation — Playwright-based browser engine.

Provides:
- Navigate, click, type, screenshot
- Accessibility tree extraction
- Console log capture
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConsoleMessage:
    """A console message from the browser.

    Attributes:
        type: Message type (log, warn, error, info).
        text: Message text.
        timestamp: When the message was captured.
    """

    type: str
    text: str
    timestamp: float


class BrowserEngine:
    """Playwright-based browser automation engine.

    Provides high-level browser interaction: navigation, clicking,
    typing, screenshots, and accessibility tree extraction.

    Note: This module is designed to work with the Playwright library.
    If Playwright is not installed, methods will raise RuntimeError
    with installation instructions.
    """

    def __init__(
        self,
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        timeout: int = 30000,
    ) -> None:
        """Initialize the browser engine.

        Args:
            headless: Run browser in headless mode (no visible window).
            viewport_width: Browser viewport width in pixels.
            viewport_height: Browser viewport height in pixels.
            timeout: Default timeout for operations (milliseconds).
        """
        self._headless = headless
        self._viewport_width = viewport_width
        self._viewport_height = viewport_height
        self._timeout = timeout
        self._browser = None
        self._context = None
        self._page = None
        self._console_messages: list[ConsoleMessage] = []
        self._closed = True

    def navigate(self, url: str) -> dict[str, Any]:
        """Navigate to a URL.

        Args:
            url: URL to navigate to.

        Returns:
            Dict with page title, URL, and load status.

        Raises:
            RuntimeError: If navigation fails.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise RuntimeError(
                "Playwright is not installed. "
                "Install it with: pip install playwright && playwright install"
            )

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self._headless)
            context = browser.new_context(
                viewport={"width": self._viewport_width, "height": self._viewport_height}
            )
            page = context.new_page()

            # Capture console messages
            page.on("console", self._handle_console)

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=self._timeout)

                return {
                    "url": page.url,
                    "title": page.title(),
                    "status": "loaded",
                }
            except Exception as e:
                return {
                    "url": url,
                    "title": "",
                    "status": f"error: {e}",
                }
            finally:
                browser.close()

    def click(self, selector: str) -> dict[str, Any]:
        """Click an element by CSS selector.

        Args:
            selector: CSS selector for the element to click.

        Returns:
            Dict with action result.
        """
        return self._execute_action(f"click '{selector}'", lambda page: page.click(selector))

    def type_text(self, selector: str, text: str) -> dict[str, Any]:
        """Type text into an element by CSS selector.

        Args:
            selector: CSS selector for the input element.
            text: Text to type.

        Returns:
            Dict with action result.
        """
        return self._execute_action(
            f"type '{text}' into '{selector}'",
            lambda page: page.fill(selector, text),
        )

    def press_key(self, key: str) -> dict[str, Any]:
        """Press a keyboard key.

        Args:
            key: Key name (Enter, Tab, Escape, ArrowDown, etc.).

        Returns:
            Dict with action result.
        """
        return self._execute_action(f"press '{key}'", lambda page: page.keyboard.press(key))

    def screenshot(self, path: str | None = None) -> dict[str, Any]:
        """Take a screenshot of the current page.

        Args:
            path: Optional file path to save the screenshot.

        Returns:
            Dict with screenshot info.
        """
        return self._execute_action("screenshot", lambda page: page.screenshot(path=path or None))

    def get_accessibility_tree(self) -> str:
        """Extract the accessibility tree as text.

        Returns:
            Text representation of the accessibility tree.
        """
        return self._execute_action(
            "accessibility tree",
            lambda page: self._extract_accessibility_tree(page),
        )

    def get_console_output(self) -> list[dict[str, Any]]:
        """Get captured console messages.

        Returns:
            List of console message dicts.
        """
        return [
            {
                "type": msg.type,
                "text": msg.text,
                "timestamp": msg.timestamp,
            }
            for msg in self._console_messages
        ]

    def clear_console(self) -> None:
        """Clear captured console messages."""
        self._console_messages.clear()

    def get_page_content(self) -> str:
        """Get the current page HTML content.

        Returns:
            Page HTML content as string.
        """
        return self._execute_action(
            "get content",
            lambda page: page.content(),
        )

    def evaluate(self, expression: str) -> Any:
        """Evaluate JavaScript in the page context.

        Args:
            expression: JavaScript expression to evaluate.

        Returns:
            The evaluation result.
        """
        return self._execute_action(
            f"evaluate: {expression[:80]}",
            lambda page: page.evaluate(expression),
        )

    def _execute_action(self, description: str, action_fn) -> Any:
        """Execute a browser action with Playwright.

        Args:
            description: Human-readable action description.
            action_fn: Function that takes a page object.

        Returns:
            Result from the action function.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise RuntimeError(
                "Playwright is not installed. "
                "Install it with: pip install playwright && playwright install"
            )

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self._headless)
            context = browser.new_context(
                viewport={"width": self._viewport_width, "height": self._viewport_height}
            )
            page = context.new_page()
            page.on("console", self._handle_console)

            try:
                result = action_fn(page)
                return {"status": "success", "result": result}
            except Exception as e:
                return {"status": "error", "error": str(e)}
            finally:
                browser.close()

    def _extract_accessibility_tree(self, page) -> str:
        """Extract accessibility tree from the page.

        Args:
            page: Playwright page object.

        Returns:
            Text representation of the accessibility tree.
        """
        try:
            tree = page.accessibility.snapshot(interesting_only=True)
            return json.dumps(tree, indent=2, default=str)
        except Exception:
            # Fallback: extract text content
            return page.inner_text("body")

    def _handle_console(self, msg) -> None:
        """Handle console messages from the browser.

        Args:
            msg: Playwright console message object.
        """
        self._console_messages.append(
            ConsoleMessage(
                type=msg.type,
                text=msg.text,
                timestamp=time.time(),
            )
        )

    def close(self) -> None:
        """Close the browser session."""
        self._closed = True
