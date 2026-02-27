from __future__ import annotations

import logging
import textwrap
from typing import Any

from src.models.types import ContentItem, Thread
from src.processors.summarizer import Summarizer

logger = logging.getLogger(__name__)

CHAR_LIMIT = 280


class ThreadGenerator:
    def __init__(self, summarizer: Summarizer | None = None, settings: Any = None) -> None:
        self.summarizer = summarizer or Summarizer()
        self.settings = settings

    def generate(self, item: ContentItem, domain: str = "geopolitics") -> Thread:
        try:
            title = item.title or "Untitled"
            source = item.source or "Unknown"
            content = (item.content or title)[:10_000]

            summary = self.summarizer.summarize(content)
            key_points = self.summarizer.extract_key_points(content, num_points=5)
            context_lines = self._build_context(content, domain)

            raw_parts: list[str] = []

            raw_parts.append(
                f"Breaking: {title[:80]} #{domain.capitalize()} {item.url}"
            )

            raw_parts.append(f"{source} reports: {summary}")

            for ctx in context_lines:
                raw_parts.append(f"Context: {ctx}")

            for point in key_points:
                raw_parts.append(f"Key Point: {point}")

            raw_parts.append(
                f"What's your take on {title[:50]}? Share below! #{domain.capitalize()}"
            )

            posts = self._format_numbered_posts(raw_parts)
            return Thread(posts=posts, platform="x", domain=domain)
        except Exception as exc:
            logger.exception("Error generating thread: %s", exc)
            fallback = f"Thread: {item.title[:50]} failed. Source: {item.source}."
            return Thread(posts=[fallback[:CHAR_LIMIT]], platform="x", domain=domain)

    def _build_context(self, content: str, domain: str) -> list[str]:
        if self.settings is None:
            return []

        domains_config = getattr(self.settings, "domains", {})
        domain_info = domains_config.get(domain, {})
        context_map: dict[str, list[str]] = domain_info.get("context", {})
        if not context_map:
            return []

        content_lower = content.lower()
        matched: list[str] = []
        for context_name, keywords in context_map.items():
            if any(kw in content_lower for kw in keywords):
                label = context_name.replace("_", " ").title()
                matched.append(label)

        return matched[:2]

    def _format_numbered_posts(self, parts: list[str]) -> list[str]:
        expanded: list[str] = []
        for part in parts:
            if len(part) <= CHAR_LIMIT - 10:
                expanded.append(part)
            else:
                wrapped = textwrap.wrap(part, width=CHAR_LIMIT - 10)
                expanded.extend(wrapped)

        total = len(expanded)
        posts: list[str] = []
        for idx, text in enumerate(expanded, start=1):
            suffix = f" ({idx}/{total})"
            available = CHAR_LIMIT - len(suffix)
            truncated = text[:available]
            posts.append(f"{truncated}{suffix}")

        return posts
