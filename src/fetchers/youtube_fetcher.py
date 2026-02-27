from __future__ import annotations

import logging
from datetime import datetime, timezone

from googleapiclient.discovery import build
from tenacity import retry, stop_after_attempt, wait_exponential
from youtube_transcript_api import YouTubeTranscriptApi

from src.models.types import ContentItem
from src.fetchers.utils import detect_and_translate, is_relevant, is_sensitive

logger = logging.getLogger(__name__)

CHANNEL_NAME_MAP: dict[str, str] = {
    "UCSiDGb0MnHFGjs5UbhZ1Qaw": "Lex Fridman",
    "UC2rGfsVex4dgKJBvFzUh-Lg": "BeerBiceps",
    "UCrC8mOqJQpoB7NuIMKIS6rQ": "Joe Rogan",
    "UCSYMy1wJ0gtM3HLoC4SnuRw": "Abhijit Chavda",
    "UC3cU0KXMBOKh3gK-2U8iHqw": "Vikas Divyakirti",
    "UCt-ybO9Kw9QqG9Ts_YaPJgA": "Nitish Rajput",
    "UCsT0YIqwnpJCM-mx7-gSA4Q": "Pavneet Singh",
}


class YouTubeFetcher:

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._youtube = build("youtube", "v3", developerKey=settings.youtube_api_key)
        try:
            self._youtube.videos().list(part="snippet", id="RY9HFhHYrZQ").execute()
            logger.info("YouTube API client initialised successfully")
        except Exception as exc:
            logger.error("YouTube API verification failed: %s", exc)
            raise

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
        transcript_warnings: set[str] = set()

        for channel_id in self._settings.youtube_channels:
            try:
                logger.info("Fetching latest video for channel %s", channel_id)
                search_response = self._youtube.search().list(
                    part="snippet",
                    channelId=channel_id,
                    maxResults=1,
                    order="date",
                ).execute()

                for result in search_response.get("items", []):
                    video_id = result["id"].get("videoId")
                    if not video_id:
                        continue

                    snippet = result["snippet"]
                    title = snippet.get("title", "")
                    description = snippet.get("description", "")

                    content, transcript_data = self._get_transcript(
                        video_id, title, description, supported_langs, channel_id, transcript_warnings
                    )

                    content, lang = detect_and_translate(content, supported_langs)

                    if not is_relevant(content, domain_keywords, breaking_keywords):
                        logger.debug("Video %s not relevant, skipping", video_id)
                        continue

                    if is_sensitive(content, sensitive_keywords):
                        logger.debug("Video %s flagged as sensitive, skipping", video_id)
                        continue

                    source_name = CHANNEL_NAME_MAP.get(channel_id, "Unknown")
                    published_at = snippet.get("publishedAt", "")
                    try:
                        timestamp = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        timestamp = datetime.now(tz=timezone.utc)

                    items.append(
                        ContentItem(
                            id=video_id,
                            title=title,
                            content=content[:10000],
                            url=f"https://www.youtube.com/watch?v={video_id}",
                            timestamp=timestamp,
                            source=source_name,
                            language=lang,
                            domain="geopolitics",
                            content_type="youtube",
                            transcript_data=transcript_data,
                        )
                    )

            except Exception as exc:
                if "quotaExceeded" in str(exc):
                    logger.error("YouTube API quota exceeded, aborting fetch")
                    return items
                logger.error("Error fetching channel %s: %s", channel_id, exc)

        logger.info("YouTubeFetcher returning %d items", len(items))
        return items

    def _get_transcript(
        self,
        video_id: str,
        title: str,
        description: str,
        supported_langs: list[str],
        channel_id: str,
        transcript_warnings: set[str],
    ) -> tuple[str, list[dict]]:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=supported_langs)
            content = " ".join(entry["text"] for entry in transcript)
            return content, transcript
        except Exception as exc:
            if channel_id not in transcript_warnings:
                logger.warning("No transcript for channel %s: %s", channel_id, str(exc)[:100])
                transcript_warnings.add(channel_id)

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
            logger.debug("No comments for video %s: %s", video_id, str(exc)[:100])

        return content, []
