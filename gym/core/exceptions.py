"""
Exception definitions shared across the testall package.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TestSkipException(Exception):
    """
    Raised when a test case must be skipped (e.g., missing resources).

    Attributes:
        reason: Human readable reason for the skip.
        details: Optional metadata that provides extra context (case id, etc.).
    """

    reason: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.reason)

    def with_detail(self, key: str, value: Any) -> "TestSkipException":
        """Fluent helper to append extra detail information."""
        self.details[key] = value
        return self
