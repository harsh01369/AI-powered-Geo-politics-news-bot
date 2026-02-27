from __future__ import annotations

import logging

import tweepy
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models.types import Thread

logger = logging.getLogger(__name__)


class TwitterPublisher:
    def __init__(self, client: tweepy.Client) -> None:
        self._client = client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def post_thread(self, thread: Thread) -> list[str]:
        tweet_ids: list[str] = []
        parent_id: str | None = None

        for post in thread.posts:
            text = post.strip()
            if not text:
                continue

            if parent_id is not None:
                response = self._client.create_tweet(
                    text=text, in_reply_to_tweet_id=parent_id
                )
            else:
                response = self._client.create_tweet(text=text)

            tweet_id = str(response.data["id"])
            tweet_ids.append(tweet_id)
            parent_id = tweet_id
            logger.info("Posted tweet %s: %s", tweet_id, text[:50])

        return tweet_ids

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def post_single(self, content: str) -> str:
        response = self._client.create_tweet(text=content)
        tweet_id = str(response.data["id"])
        logger.info("Posted single tweet %s: %s", tweet_id, content[:50])
        return tweet_id
