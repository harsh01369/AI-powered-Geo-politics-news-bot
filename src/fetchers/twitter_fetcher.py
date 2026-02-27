from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import tweepy
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models.types import ContentItem
from src.fetchers.utils import is_relevant, is_sensitive

logger = logging.getLogger(__name__)


class TwitterFetcher:

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._client = tweepy.Client(
            consumer_key=settings.x_api_key,
            consumer_secret=settings.x_api_secret,
            access_token=settings.x_access_token,
            access_token_secret=settings.x_access_token_secret,
        )
        try:
            me = self._client.get_me()
            if me.data is None:
                raise tweepy.TweepyException("get_me() returned no data")
            self._user = me.data
            logger.info("Twitter API authenticated as @%s", self._user.username)
        except tweepy.TweepyException as exc:
            if hasattr(exc, "response") and exc.response is not None:
                if exc.response.status_code == 401:
                    logger.error("Twitter 401 Unauthorized. Verify API credentials.")
                elif exc.response.status_code == 429:
                    logger.error("Twitter 429 Rate-limit hit during init.")
            logger.error("Failed to initialise Twitter client: %s", exc)
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

        for account in self._settings.x_accounts:
            try:
                logger.info("Fetching tweets for account: %s", account)
                user_resp = self._client.get_user(username=account)
                if user_resp.data is None:
                    logger.warning("Account '%s' not found, skipping", account)
                    continue
                user_id = user_resp.data.id

                tweets = self._client.get_users_tweets(
                    id=user_id,
                    tweet_fields=["created_at", "text"],
                    max_results=10,
                )
                if not tweets.data:
                    logger.info("No tweets returned for @%s", account)
                    continue

                logger.info("Fetched %d tweets from @%s", len(tweets.data), account)

                for tweet in tweets.data:
                    text = tweet.text

                    if not is_relevant(text, domain_keywords, breaking_keywords):
                        logger.debug("Tweet %s not relevant, skipping", tweet.id)
                        continue

                    if is_sensitive(text, sensitive_keywords):
                        logger.debug("Tweet %s flagged as sensitive, skipping", tweet.id)
                        continue

                    items.append(
                        ContentItem(
                            id=str(tweet.id),
                            title=text[:120],
                            content=text,
                            url=f"https://x.com/{account}/status/{tweet.id}",
                            timestamp=tweet.created_at or datetime.now(tz=timezone.utc),
                            source=account,
                            language="en",
                            domain="geopolitics",
                            content_type="tweet",
                        )
                    )

            except tweepy.TweepyException as exc:
                if hasattr(exc, "response") and exc.response is not None:
                    if exc.response.status_code == 401:
                        logger.error("401 Unauthorized when fetching @%s. Check credentials.", account)
                    elif exc.response.status_code == 429:
                        logger.error("429 Rate-limit exceeded for @%s. Backing off 15 min.", account)
                        time.sleep(900)
                else:
                    logger.error("Twitter API error for @%s: %s", account, exc)
                continue

        logger.info("TwitterFetcher returning %d items", len(items))
        return items
