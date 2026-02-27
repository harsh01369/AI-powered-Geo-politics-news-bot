from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

CONFIG_FILE = Path("newsbot_config.json")

REQUIRED_ENV_VARS = [
    "X_API_KEY",
    "X_API_SECRET",
    "X_ACCESS_TOKEN",
    "X_ACCESS_TOKEN_SECRET",
    "YOUTUBE_API_KEY",
]

DEFAULT_CONFIG: dict[str, Any] = {
    "x_accounts": ["ArmMonitor11"],
    "youtube_channels": [
        "UCsT0YIqwnpJCM-mx7-gSA4Q",
        "UCt-ybO9Kw9QqG9Ts_YaPJgA",
        "UC3cU0KXMBOKh3gK-2U8iHqw",
        "UCSYMy1wJ0gtM3HLoC4SnuRw",
        "UCrC8mOqJQpoB7NuIMKIS6rQ",
        "UC2rGfsVex4dgKJBvFzUh-Lg",
        "UCSiDGb0MnHFGjs5UbhZ1Qaw",
    ],
    "rss_feeds": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://feeds.reuters.com/reuters/topNews",
        "https://www.aljazeera.com/xml/rss/all.xml",
        "https://www.thehindu.com/news/international/feeder/default.rss",
        "https://apnews.com/hub/world-news/rss",
    ],
    "sensitive_keywords": [
        "violence",
        "hate",
        "graphic",
        "terrorism",
        "explicit",
        "riot",
        "radicalization",
        "communal",
    ],
    "relevance_keywords": [
        "politics",
        "geopolitics",
        "diplomacy",
        "policy",
        "government",
        "trade",
        "conflict",
        "summit",
        "election",
        "sanction",
        "war",
        "treaty",
        "alliance",
    ],
    "breaking_keywords": [
        "breaking",
        "urgent",
        "summit",
        "conflict",
        "crisis",
        "announce",
        "declare",
    ],
    "schedule_interval_minutes": 15,
    "supported_languages": [
        "en",
        "hi",
        "es",
        "fr",
        "de",
        "ar",
        "ru",
        "pt",
        "zh-Hans",
        "zh-Hant",
        "it",
        "ja",
    ],
    "test_mode": False,
    "domains": {
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
                "israel_iran": [
                    "israel",
                    "iran",
                    "hezbollah",
                    "hamas",
                    "nuclear deal",
                    "middle east",
                    "idf",
                    "irgc",
                ],
                "nato": [
                    "nato",
                    "north atlantic",
                    "article 5",
                    "collective defense",
                    "military alliance",
                ],
                "us_china": [
                    "us china",
                    "taiwan",
                    "south china sea",
                    "trade war",
                    "tariff",
                    "semiconductor",
                    "indo-pacific",
                ],
                "india_pakistan": [
                    "india pakistan",
                    "kashmir",
                    "loc",
                    "line of control",
                    "south asia",
                ],
                "russia_ukraine": [
                    "russia",
                    "ukraine",
                    "crimea",
                    "donbas",
                    "zelensky",
                    "putin",
                    "black sea",
                ],
            },
        },
        "spiritual": {
            "keywords": [
                "spiritual",
                "meditation",
                "consciousness",
                "mindfulness",
                "enlightenment",
                "yoga",
                "dharma",
                "faith",
                "prayer",
            ],
            "context": {},
        },
    },
}


@dataclass
class Settings:
    """Holds all application configuration loaded from environment variables and config file."""

    x_api_key: str
    x_api_secret: str
    x_access_token: str
    x_access_token_secret: str
    youtube_api_key: str
    x_accounts: list[str] = field(default_factory=list)
    youtube_channels: list[str] = field(default_factory=list)
    rss_feeds: list[str] = field(default_factory=list)
    sensitive_keywords: list[str] = field(default_factory=list)
    relevance_keywords: list[str] = field(default_factory=list)
    breaking_keywords: list[str] = field(default_factory=list)
    schedule_interval_minutes: int = 15
    supported_languages: list[str] = field(default_factory=list)
    test_mode: bool = False
    domains: dict[str, Any] = field(default_factory=dict)


def _load_config_file() -> dict[str, Any]:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, encoding="utf-8") as f:
            user_config = json.load(f)
        merged = {**DEFAULT_CONFIG, **user_config}
        return merged
    return dict(DEFAULT_CONFIG)


def _validate_env_vars() -> dict[str, str]:
    env_values: dict[str, str] = {}
    missing: list[str] = []
    for var in REQUIRED_ENV_VARS:
        value = os.getenv(var)
        if not value:
            missing.append(var)
        else:
            env_values[var] = value
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Please set them in your .env file or system environment."
        )
    return env_values


def load_settings() -> Settings:
    env_values = _validate_env_vars()
    config = _load_config_file()

    return Settings(
        x_api_key=env_values["X_API_KEY"],
        x_api_secret=env_values["X_API_SECRET"],
        x_access_token=env_values["X_ACCESS_TOKEN"],
        x_access_token_secret=env_values["X_ACCESS_TOKEN_SECRET"],
        youtube_api_key=env_values["YOUTUBE_API_KEY"],
        x_accounts=config.get("x_accounts", DEFAULT_CONFIG["x_accounts"]),
        youtube_channels=config.get("youtube_channels", DEFAULT_CONFIG["youtube_channels"]),
        rss_feeds=config.get("rss_feeds", DEFAULT_CONFIG["rss_feeds"]),
        sensitive_keywords=config.get("sensitive_keywords", DEFAULT_CONFIG["sensitive_keywords"]),
        relevance_keywords=config.get("relevance_keywords", DEFAULT_CONFIG["relevance_keywords"]),
        breaking_keywords=config.get("breaking_keywords", DEFAULT_CONFIG["breaking_keywords"]),
        schedule_interval_minutes=config.get(
            "schedule_interval_minutes", DEFAULT_CONFIG["schedule_interval_minutes"]
        ),
        supported_languages=config.get(
            "supported_languages", DEFAULT_CONFIG["supported_languages"]
        ),
        test_mode=config.get("test_mode", DEFAULT_CONFIG["test_mode"]),
        domains=config.get("domains", DEFAULT_CONFIG["domains"]),
    )
