"""Tests for src.models.types dataclasses."""
from __future__ import annotations

from datetime import datetime

from src.models.types import ContentItem, Claim, Thread, VerificationResult


class TestContentItem:
    def test_creation_with_all_fields(self):
        item = ContentItem(
            id="item-1",
            title="Summit Report",
            content="Detailed analysis of the summit.",
            url="https://example.com/report",
            timestamp=datetime(2025, 1, 10, 8, 30, 0),
            source="BBC",
            language="en",
            domain="geopolitics",
            content_type="news",
            transcript_data=[{"text": "Hello", "start": 0, "duration": 5}],
        )
        assert item.id == "item-1"
        assert item.title == "Summit Report"
        assert item.content == "Detailed analysis of the summit."
        assert item.url == "https://example.com/report"
        assert item.timestamp == datetime(2025, 1, 10, 8, 30, 0)
        assert item.source == "BBC"
        assert item.language == "en"
        assert item.domain == "geopolitics"
        assert item.content_type == "news"
        assert item.transcript_data == [{"text": "Hello", "start": 0, "duration": 5}]

    def test_defaults(self):
        item = ContentItem(
            id="item-2",
            title="Title",
            content="Content",
            url="https://example.com",
            timestamp=datetime.now(),
            source="Test",
            language="en",
            domain="geopolitics",
            content_type="news",
        )
        assert item.transcript_data is None


class TestThread:
    def test_creation(self):
        thread = Thread(
            posts=["Post 1", "Post 2"],
            platform="x",
            domain="geopolitics",
        )
        assert thread.posts == ["Post 1", "Post 2"]
        assert thread.platform == "x"
        assert thread.domain == "geopolitics"

    def test_defaults(self):
        thread = Thread()
        assert thread.posts == []
        assert thread.platform == ""
        assert thread.domain == ""


class TestVerificationResult:
    def test_creation(self):
        result = VerificationResult(
            verified=True,
            sources=["https://bbc.com/article"],
            confidence=0.85,
        )
        assert result.verified is True
        assert result.sources == ["https://bbc.com/article"]
        assert result.confidence == 0.85

    def test_defaults(self):
        result = VerificationResult()
        assert result.verified is False
        assert result.sources == []
        assert result.confidence == 0.0


class TestClaim:
    def test_creation(self):
        now = datetime.now()
        claim = Claim(
            id="claim-1",
            source="Reuters",
            claim_text="World leaders met at the summit.",
            status="pending",
            timestamp=now,
            content_type="news",
            post_content="Breaking: summit announced",
            confidence=0.75,
        )
        assert claim.id == "claim-1"
        assert claim.source == "Reuters"
        assert claim.claim_text == "World leaders met at the summit."
        assert claim.status == "pending"
        assert claim.timestamp == now
        assert claim.content_type == "news"
        assert claim.post_content == "Breaking: summit announced"
        assert claim.confidence == 0.75
