"""Adversarial Protection — input sanitization, prompt injection detection, command safety.

Provides:
- Prompt injection detection in retrieved content
- Input sanitization
- Command allowlisting/blocklisting for sensitive ops
- Rate limiting on tool calls
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any


# Patterns that commonly indicate prompt injection attempts
INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+all\s+previous\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+all\s+previous\s+instructions?", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"system\s*:?.*?instructions?", re.IGNORECASE),
    re.compile(r"<!\[CDATA\[.*?\]\]", re.IGNORECASE | re.DOTALL),
    re.compile(r"<script\b", re.IGNORECASE),
    re.compile(r"```(?:xml|xmltag|systemprompt)", re.IGNORECASE),
    re.compile(r"begin\s+of\s+secret", re.IGNORECASE),
    re.compile(r"\[INST\]|\[/INST\]", re.IGNORECASE),
    re.compile(r"your\s+(?:new\s+)?task\s*(?:is|:)", re.IGNORECASE),
    re.compile(r"pretend\s+to\s+be\s+", re.IGNORECASE),
    re.compile(r"act\s+as\s+if\s+", re.IGNORECASE),
]

# Dangerous shell commands that should be blocklisted
BLOCKED_COMMANDS: list[str] = [
    r"rm\s+-rf\s+/",
    r"rm\s+-rf\s+/\*",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r":\s*\(\s*\)\s*\{",  # fork bomb start
    r"curl\s+.*\|\s*bash",
    r"wget\s+.*\|\s*bash",
    r"chmod\s+-R\s+777\s+/",
    r"chown\s+-R\s+\.\s+/",
]


@dataclass
class SafetyResult:
    """Result of a safety check.

    Attributes:
        safe: Whether the input is considered safe.
        threats: List of detected threat descriptions.
        action: Recommended action (allow, block, warn, sanitize).
    """

    safe: bool
    threats: list[str] = field(default_factory=list)
    action: str = "allow"


class PromptInjectionDetector:
    """Detect potential prompt injection in text content.

    Scans text for patterns commonly used in prompt injection attacks.
    """

    def __init__(self, patterns: list[re.Pattern] | None = None) -> None:
        self._patterns = patterns or INJECTION_PATTERNS

    def check(self, text: str) -> SafetyResult:
        """Check text for prompt injection patterns.

        Args:
            text: Text to scan.

        Returns:
            SafetyResult with detection info.
        """
        threats: list[str] = []

        for pattern in self._patterns:
            match = pattern.search(text)
            if match:
                threats.append(
                    f"Possible prompt injection: '{match.group()[:50]}'"
                )

        if threats:
            return SafetyResult(
                safe=False,
                threats=threats,
                action="warn",
            )

        return SafetyResult(safe=True, action="allow")

    def sanitize(self, text: str) -> str:
        """Sanitize text by escaping potential injection patterns.

        Wraps suspicious segments with escape markers so they are treated
        as data rather than instructions.

        Args:
            text: Raw text to sanitize.

        Returns:
            Sanitized text with injection patterns neutralized.
        """
        result = text

        # Escape system-style markers
        result = result.replace("[INST]", "[S-INJECT-SB]INST[S-INJECT-EB]")
        result = result.replace("[/INST]", "[S-INJECT-SB]/INST[S-INJECT-EB]")

        # Wrap CDATA sections
        result = re.sub(
            r"<!\[CDATA\[.*?\]\]>",
            lambda m: f"[sanitized-data]{m.group()[:80]}...",
            result,
            flags=re.DOTALL,
        )

        # Escape XML system prompt tags
        result = re.sub(
            r"```(?:xml|xmltag|systemprompt)",
            "```[sanitized]",
            result,
            flags=re.IGNORECASE,
        )

        return result


class CommandSafetyChecker:
    """Check shell commands for dangerous operations.

    Uses blocklist and allowlist patterns to determine if a command
    should be executed.
    """

    def __init__(
        self,
        blocklist: list[str] | None = None,
        allowlist: list[str] | None = None,
    ) -> None:
        self._block_patterns = [
            re.compile(p, re.IGNORECASE) for p in (blocklist or BLOCKED_COMMANDS)
        ]
        self._allow_patterns = [
            re.compile(p, re.IGNORECASE) for p in (allowlist or [])
        ]

    def check(self, command: str) -> SafetyResult:
        """Check if a command is safe to execute.

        Args:
            command: Shell command string.

        Returns:
            SafetyResult with the safety verdict.
        """
        # If allowlist is set, command must match at least one pattern
        if self._allow_patterns:
            allowed = any(p.search(command) for p in self._allow_patterns)
            if not allowed:
                return SafetyResult(
                    safe=False,
                    threats=[f"Command not in allowlist: '{command[:80]}'"],
                    action="block",
                )

        # Check blocklist
        for pattern in self._block_patterns:
            if pattern.search(command):
                return SafetyResult(
                    safe=False,
                    threats=[f"Blocked command pattern: '{pattern.pattern}'"],
                    action="block",
                )

        # Check for obviously dangerous patterns
        dangerous = self._check_dangerous_patterns(command)
        if dangerous:
            return SafetyResult(
                safe=False,
                threats=dangerous,
                action="block",
            )

        return SafetyResult(safe=True, action="allow")

    def _check_dangerous_patterns(self, command: str) -> list[str]:
        """Check for patterns that are almost always dangerous."""
        threats: list[str] = []

        # Redirect to /dev/null hiding output + destructive ops
        if ">/dev/null 2>&1" in command and ("rm " in command or "dd " in command):
            threats.append("Command hides output while performing destructive operation")

        # Eval or exec of variables
        if re.search(r"\beval\s+\$", command):
            threats.append("eval of variable detected - potential code injection")

        # Sudo usage
        if command.strip().startswith("sudo"):
            threats.append("sudo usage detected")

        return threats


class RateLimiter:
    """Rate limiter for tool calls.

    Prevents rapid-fire tool calls that could overwhelm the system
    or the LLM provider.
    """

    def __init__(
        self,
        max_calls_per_minute: int = 60,
        max_calls_per_hour: int = 1000,
    ) -> None:
        self._max_per_minute = max_calls_per_minute
        self._max_per_hour = max_calls_per_hour
        self._timestamps: list[float] = []

    def check(self) -> SafetyResult:
        """Check if a new tool call is within rate limits.

        Returns:
            SafetyResult indicating if the call should proceed.
        """
        now = time.time()
        self._cleanup(now)

        # Check per-minute limit
        recent = [t for t in self._timestamps if now - t < 60]
        if len(recent) >= self._max_per_minute:
            return SafetyResult(
                safe=False,
                threats=[
                    f"Rate limit exceeded: {len(recent)} calls in last minute "
                    f"(max {self._max_per_minute})"
                ],
                action="block",
            )

        # Check per-hour limit
        hourly = [t for t in self._timestamps if now - t < 3600]
        if len(hourly) >= self._max_per_hour:
            return SafetyResult(
                safe=False,
                threats=[
                    f"Hourly rate limit exceeded: {len(hourly)} calls "
                    f"(max {self._max_per_hour})"
                ],
                action="block",
            )

        return SafetyResult(safe=True, action="allow")

    def record_call(self) -> None:
        """Record a tool call timestamp."""
        self._timestamps.append(time.time())

    def _cleanup(self, now: float) -> None:
        """Remove timestamps older than 1 hour."""
        self._timestamps = [t for t in self._timestamps if now - t < 3600]

    @property
    def calls_in_last_minute(self) -> int:
        """Number of calls in the last minute."""
        now = time.time()
        return len([t for t in self._timestamps if now - t < 60])

    @property
    def calls_in_last_hour(self) -> int:
        """Number of calls in the last hour."""
        now = time.time()
        return len([t for t in self._timestamps if now - t < 3600])


class SafetyGuard:
    """Unified safety guard combining all checks.

    Provides a single entry point for comprehensive safety checks.
    """

    def __init__(
        self,
        injection_detector: PromptInjectionDetector | None = None,
        command_checker: CommandSafetyChecker | None = None,
        rate_limiter: RateLimiter | None = None,
        rate_limit_enabled: bool = True,
    ) -> None:
        self._injection = injection_detector or PromptInjectionDetector()
        self._command = command_checker or CommandSafetyChecker()
        self._rate = rate_limiter if rate_limit_enabled else None
        if rate_limit_enabled and rate_limiter is None:
            self._rate = RateLimiter()

    def check_input(self, text: str) -> SafetyResult:
        """Check user input for safety.

        Args:
            text: User input text.

        Returns:
            SafetyResult.
        """
        return self._injection.check(text)

    def check_command(self, command: str) -> SafetyResult:
        """Check a shell command for safety.

        Args:
            command: Shell command string.

        Returns:
            SafetyResult.
        """
        return self._command.check(command)

    def check_retrieved_content(self, content: str) -> SafetyResult:
        """Check RAG-retrieved content for prompt injection.

        Args:
            content: Retrieved document content.

        Returns:
            SafetyResult.
        """
        result = self._injection.check(content)
        if not result.safe:
            # For retrieved content, sanitize instead of blocking
            result.action = "sanitize"
        return result

    def check_rate_limit(self) -> SafetyResult:
        """Check if the current call is within rate limits.

        Returns:
            SafetyResult.
        """
        if self._rate is None:
            return SafetyResult(safe=True, action="allow")
        return self._rate.check()

    def record_call(self) -> None:
        """Record a tool call for rate limiting."""
        if self._rate is not None:
            self._rate.record_call()

    def sanitize_content(self, content: str) -> str:
        """Sanitize content to neutralize injection patterns.

        Args:
            content: Raw content.

        Returns:
            Sanitized content.
        """
        return self._injection.sanitize(content)
