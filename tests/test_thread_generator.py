"""Tests for src.generators.thread_generator.ThreadGenerator.

The Summarizer is mocked so no heavy ML model is required.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from src.generators.thread_generator import CHAR_LIMIT, ThreadGenerator
from src.models.types import ContentItem


def _make_content_item() -> ContentItem:
    return ContentItem(
        id="gen-001",
        title="UN Summit on Global Sanctions",
        content=(
            "World leaders gathered at the UN to discuss new sanctions "
            "against several nations. The summit addressed trade policies, "
            "military alliances, and diplomatic relations across continents."
        ),
        url="https://example.com/summit",
        timestamp=datetime(2025, 6, 20, 9, 0, 0),
        source="Reuters",
        language="en",
        domain="geopolitics",
        content_type="news",
    )


def _make_mock_summarizer() -> MagicMock:
    mock = MagicMock()
    mock.summarize.return_value = "Leaders discuss sanctions at UN summit"
    mock.extract_key_points.return_value = [
        "New sanctions proposed",
        "Trade policies reviewed",
        "Military alliances strengthened",
    ]
    return mock


class TestThreadPostCount:
    def test_generates_thread_with_correct_post_count(self):
        """Thread should contain between 2 and 7 posts (inclusive)."""
        summarizer = _make_mock_summarizer()
        generator = ThreadGenerator(summarizer=summarizer)
        thread = generator.generate(_make_content_item(), domain="geopolitics")

        assert len(thread.posts) >= 2
        assert len(thread.posts) <= 7


class TestThreadCharLimit:
    def test_posts_respect_280_char_limit(self):
        """Every post in the generated thread must be at most 280 characters."""
        summarizer = _make_mock_summarizer()
        generator = ThreadGenerator(summarizer=summarizer)
        thread = generator.generate(_make_content_item(), domain="geopolitics")

        for i, post in enumerate(thread.posts):
            assert len(post) <= CHAR_LIMIT, (
                f"Post {i} exceeds {CHAR_LIMIT} chars (len={len(post)}): {post!r}"
            )


class TestThreadIncludesSource:
    def test_thread_includes_source(self):
        """At least one post in the thread should mention the content source."""
        summarizer = _make_mock_summarizer()
        generator = ThreadGenerator(summarizer=summarizer)
        item = _make_content_item()
        thread = generator.generate(item, domain="geopolitics")

        full_text = " ".join(thread.posts)
        assert item.source in full_text, (
            f"Expected source '{item.source}' to appear in thread posts"
        )
