from __future__ import annotations

import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.models.types import ContentItem
from src.storage.database import Database


@pytest.fixture
def sample_content_item() -> ContentItem:
    """A fully-populated ContentItem for use in tests."""
    return ContentItem(
        id="test-001",
        title="Breaking: Major Geopolitical Summit Announced",
        content="World leaders gather to discuss sanctions and diplomacy at the UN summit.",
        url="https://example.com/news/summit",
        timestamp=datetime(2025, 6, 15, 12, 0, 0),
        source="Reuters",
        language="en",
        domain="geopolitics",
        content_type="news",
        transcript_data=None,
    )


@pytest.fixture
def sample_settings() -> MagicMock:
    """A mock settings object with test configuration values."""
    settings = MagicMock()
    settings.domains = {
        "geopolitics": {
            "keywords": [
                "geopolitics",
                "diplomacy",
                "sanctions",
                "military",
                "alliance",
                "treaty",
                "conflict",
                "war",
                "nato",
                "united nations",
            ],
            "context": {
                "russia_ukraine": ["russia", "ukraine", "crimea", "donbas"],
                "us_china": ["us china", "taiwan", "south china sea", "tariff"],
            },
        },
    }
    settings.breaking_keywords = [
        "breaking",
        "urgent",
        "summit",
        "conflict",
        "crisis",
    ]
    settings.sensitive_keywords = [
        "violence",
        "hate",
        "graphic",
        "terrorism",
        "explicit",
    ]
    settings.relevance_keywords = [
        "politics",
        "geopolitics",
        "diplomacy",
        "conflict",
    ]
    settings.supported_languages = ["en", "hi"]
    settings.test_mode = True
    return settings


@pytest.fixture
def temp_database(tmp_path):
    """A Database backed by a temporary SQLite file, cleaned up after the test."""
    db_file = str(tmp_path / "test_news_bot.db")
    db = Database(db_path=db_file)
    yield db
    db.close()
    if os.path.exists(db_file):
        os.remove(db_file)
