from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone

import feedparser
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models.types import ContentItem
from src.fetchers.utils import detect_and_translate, is_relevant, is_sensitive

logger = logging.getLogger(__name__)

_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NewsBotFetcher/1.0)"}


class RSSFetcher:

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def fetch(self) -> list[ContentItem]:
        items: list[ContentItem] = []
        domain_keywords = self._settings.domains.get("geopolitics", {}).get("keywords", [])
        breaking_keywords = self._settings.breaking_keywords
        sensitive_keywords = self._settings.sensitive_keywords
        supported_langs = self._settings.supported_languages

        for feed_url in self._settings.rss_feeds:
            try:
                logger.info("Parsing RSS feed: %s", feed_url)
                feed = feedparser.parse(feed_url)
                feed_title = feed.feed.get("title", "Unknown")

                for entry in feed.entries[:5]:
                    title = entry.get("title", "")
                    link = entry.get("link", "")
                    published = entry.get("published", "")

                    try:
                        timestamp = datetime.fromisoformat(published.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        timestamp = datetime.now(tz=timezone.utc)

                    content = self._fetch_article_text(link)
                    if not content:
                        content = entry.get("summary", title)[:1000]

                    content, lang = detect_and_translate(content, supported_langs)

                    if not is_relevant(content, domain_keywords, breaking_keywords):
                        logger.debug("RSS entry '%s' not relevant, skipping", title[:50])
                        continue

                    if is_sensitive(content, sensitive_keywords):
                        logger.debug("RSS entry '%s' flagged as sensitive, skipping", title[:50])
                        continue

                    items.append(
                        ContentItem(
                            id=str(uuid.uuid4()),
                            title=title,
                            content=content[:10000],
                            url=link,
                            timestamp=timestamp,
                            source=feed_title,
                            language=lang,
                            domain="geopolitics",
                            content_type="rss",
                        )
                    )

            except Exception as exc:
                logger.error("Error processing feed %s: %s", feed_url, exc)

        logger.info("RSSFetcher returning %d items", len(items))
        return items

    @staticmethod
    def _fetch_article_text(url: str) -> str:
        try:
            response = requests.get(url, timeout=10, headers=_HTTP_HEADERS)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")[:20]
            text = " ".join(p.get_text() for p in paragraphs if p.get_text())
            text = re.sub(r"\s+", " ", text).strip()
            return text[:10000]
        except Exception as exc:
            logger.warning("Failed to fetch article text from %s: %s", url, exc)
            return ""
