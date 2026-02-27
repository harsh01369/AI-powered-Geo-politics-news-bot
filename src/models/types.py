from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ContentItem:
    """A piece of content fetched from any source (RSS, YouTube, X/Twitter)."""

    id: str
    title: str
    content: str
    url: str
    timestamp: datetime
    source: str
    language: str
    domain: str
    content_type: str
    transcript_data: list[dict] | None = None


@dataclass
class Claim:
    """A factual claim extracted from content, pending or completed verification."""

    id: str
    source: str
    claim_text: str
    status: str
    timestamp: datetime
    content_type: str
    post_content: str
    confidence: float


@dataclass
class Thread:
    """A formatted thread of posts ready for publishing to a platform."""

    posts: list[str] = field(default_factory=list)
    platform: str = ""
    domain: str = ""


@dataclass
class VerificationResult:
    """The outcome of a fact-checking / verification process."""

    verified: bool = False
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.0
