from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from tenacity import retry, stop_after_attempt, wait_exponential
from youtube_transcript_api import YouTubeTranscriptApi

from src.models.types import ContentItem
from src.fetchers.utils import detect_and_translate

logger = logging.getLogger(__name__)

_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NewsBotFetcher/1.0)"}
_YOUTUBE_URL_PATTERN = re.compile(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})")


class URLFetcher:

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._youtube = build(
            "youtube", "v3", developerKey=settings.youtube_api_key
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def fetch(self, url: str, domain: str = "geopolitics") -> ContentItem | None:
        try:
            if "youtube.com" in url or "youtu.be" in url:
                return self._fetch_youtube(url, domain)
            return self._fetch_webpage(url, domain)
        except Exception as exc:
            logger.error("Error fetching URL %s: %s", url, exc)
            return None

    def _fetch_youtube(self, url: str, domain: str) -> ContentItem | None:
        match = _YOUTUBE_URL_PATTERN.search(url)
        if not match:
            logger.warning("Invalid YouTube URL: %s", url)
            return None

        video_id = match.group(1)
        logger.info("Fetching YouTube video metadata for %s", video_id)

        response = self._youtube.videos().list(
            part="snippet", id=video_id
        ).execute()
        video_items = response.get("items", [])
        if not video_items:
            logger.warning("No video found for ID %s", video_id)
            return None

        snippet = video_items[0].get("snippet", {})
        title = snippet.get("title", "Unknown")
        description = snippet.get("description", "")

        content, transcript_data = self._get_transcript(
            video_id, title, description
        )

        supported_langs = self._settings.supported_languages
        content, lang = detect_and_translate(content, supported_langs)

        return ContentItem(
            id=video_id,
            title=title,
            content=content[:10000],
            url=f"https://www.youtube.com/watch?v={video_id}",
            timestamp=datetime.now(tz=timezone.utc),
            source="Custom YouTube",
            language=lang,
            domain=domain,
            content_type="youtube",
            transcript_data=transcript_data,
        )

    def _fetch_webpage(self, url: str, domain: str) -> ContentItem | None:
        logger.info("Scraping webpage: %s", url)
        response = requests.get(url, timeout=30, headers=_HTTP_HEADERS)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")[:20]
        text = " ".join(p.get_text() for p in paragraphs if p.get_text())
        text = re.sub(r"\s+", " ", text).strip()[:10000]

        title = soup.title.string if soup.title else url.split("/")[-1]

        supported_langs = self._settings.supported_languages
        text, lang = detect_and_translate(text, supported_langs)

        return ContentItem(
            id=str(uuid.uuid4()),
            title=title,
            content=text,
            url=url,
            timestamp=datetime.now(tz=timezone.utc),
            source="Custom URL",
            language=lang,
            domain=domain,
            content_type="webpage",
        )

    def _get_transcript(
        self, video_id: str, title: str, description: str
    ) -> tuple[str, list[dict]]:
        supported_langs = self._settings.supported_languages
        try:
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id, languages=supported_langs
            )
            content = " ".join(entry["text"] for entry in transcript)
            return content, transcript
        except Exception as exc:
            logger.warning(
                "No transcript for video %s: %s", video_id, str(exc)[:100]
            )

        content = f"{title} {description}"
        try:
            comment_response = self._youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=3,
                textFormat="plainText",
            ).execute()
            comments = [
                item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                for item in comment_response.get("items", [])
            ]
            content += " " + " ".join(comments)
        except Exception as exc:
            logger.debug(
                "No comments for video %s: %s", video_id, str(exc)[:100]
            )

        return content, []
