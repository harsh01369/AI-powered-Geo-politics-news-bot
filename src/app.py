from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import schedule
import tweepy
from flask import Flask
from flask_cors import CORS

from src.api.routes import api
from src.config.logging import setup_logging
from src.config.settings import load_settings, Settings
from src.fetchers.twitter_fetcher import TwitterFetcher
from src.fetchers.youtube_fetcher import YouTubeFetcher
from src.generators.thread_generator import ThreadGenerator
from src.processors.summarizer import Summarizer
from src.processors.verifier import FactVerifier
from src.publishers.twitter_publisher import TwitterPublisher
from src.storage.database import Database

logger = logging.getLogger(__name__)


def create_flask_app(
    settings: Settings,
    database: Database,
    publisher: TwitterPublisher,
    generator: ThreadGenerator,
    verifier: FactVerifier,
) -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "http://localhost:8000"}})

    app.config["SETTINGS"] = settings
    app.config["DATABASE"] = database
    app.config["PUBLISHER"] = publisher
    app.config["GENERATOR"] = generator
    app.config["VERIFIER"] = verifier

    app.register_blueprint(api)
    return app


def process_content(
    settings: Settings,
    database: Database,
    twitter_fetcher: TwitterFetcher,
    youtube_fetcher: YouTubeFetcher | None,
    generator: ThreadGenerator,
    verifier: FactVerifier,
    publisher: TwitterPublisher,
    exclude_youtube: bool = False,
) -> None:
    from src.fetchers.utils import is_sensitive

    try:
        all_items = []

        try:
            x_items = twitter_fetcher.fetch()
            all_items.extend(x_items)
            database.log_metric("x_posts_fetched", len(x_items))
        except Exception as exc:
            logger.error("Error fetching X posts: %s", exc)

        if not exclude_youtube and youtube_fetcher is not None:
            try:
                yt_items = youtube_fetcher.fetch()
                all_items.extend(yt_items)
                database.log_metric("youtube_videos_fetched", len(yt_items))
            except Exception as exc:
                logger.error("Error fetching YouTube videos: %s", exc)

        for item in all_items:
            try:
                if database.is_processed(item.id):
                    logger.debug("Content %s already processed, skipping", item.id)
                    continue

                claim_text = (item.content or item.title)[:500]
                result = verifier.verify(claim_text)
                thread = generator.generate(item, item.domain)
                post_content = "\n".join(thread.posts)

                import uuid
                from datetime import datetime, timezone
                from src.models.types import Claim

                claim = Claim(
                    id=str(uuid.uuid4()),
                    source=item.source,
                    claim_text=claim_text,
                    status="pending",
                    timestamp=datetime.now(tz=timezone.utc),
                    content_type=f"{item.domain}_thread",
                    post_content=post_content,
                    confidence=result.confidence,
                )

                if result.verified and not is_sensitive(post_content, settings.sensitive_keywords):
                    if item.content_type == "news":
                        publisher.post_thread(thread)
                        claim.status = "approved"

                database.save_claim(claim)
                database.save_thread(item.url, item.domain, item.content, thread)
                database.mark_processed(item.id, item.source)
                database.log_metric("threads_generated", 1)

            except Exception as exc:
                logger.error("Error processing content %s: %s", item.id, exc)

    except Exception as exc:
        logger.error("Error in process_content: %s", exc)


async def run_scheduler(
    settings: Settings,
    database: Database,
    twitter_fetcher: TwitterFetcher,
    youtube_fetcher: YouTubeFetcher | None,
    generator: ThreadGenerator,
    verifier: FactVerifier,
    publisher: TwitterPublisher,
) -> None:
    def _process_no_youtube() -> None:
        process_content(
            settings, database, twitter_fetcher, youtube_fetcher,
            generator, verifier, publisher, exclude_youtube=True,
        )

    def _process_all() -> None:
        process_content(
            settings, database, twitter_fetcher, youtube_fetcher,
            generator, verifier, publisher, exclude_youtube=False,
        )

    schedule.every(settings.schedule_interval_minutes).minutes.do(_process_no_youtube)
    schedule.every(2).hours.do(_process_all)

    logger.info(
        "Scheduler started: X/RSS every %d min, YouTube every 2 hours",
        settings.schedule_interval_minutes,
    )

    while True:
        try:
            schedule.run_pending()
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error("Scheduler error: %s", exc)
            await asyncio.sleep(60)


async def run_flask(app: Flask) -> None:
    from wsgiref.simple_server import make_server

    server = make_server("127.0.0.1", 5001, app)
    logger.info("Flask server starting on http://127.0.0.1:5001")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, server.serve_forever)


async def main() -> None:
    setup_logging()
    logger.info("Starting news bot...")

    settings = load_settings()

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    database = Database()
    summarizer = Summarizer()
    verifier = FactVerifier()
    generator = ThreadGenerator(summarizer=summarizer, settings=settings)

    tweepy_client = tweepy.Client(
        consumer_key=settings.x_api_key,
        consumer_secret=settings.x_api_secret,
        access_token=settings.x_access_token,
        access_token_secret=settings.x_access_token_secret,
    )
    publisher = TwitterPublisher(tweepy_client)
    twitter_fetcher = TwitterFetcher(settings)

    youtube_fetcher: YouTubeFetcher | None = None
    try:
        youtube_fetcher = YouTubeFetcher(settings)
    except Exception as exc:
        logger.error("YouTube fetcher init failed, YouTube processing disabled: %s", exc)

    app = create_flask_app(settings, database, publisher, generator, verifier)

    if settings.test_mode:
        logger.info("Running in test mode")

    def _shutdown(sig: int, _frame: object) -> None:
        logger.info("Received signal %s, shutting down...", sig)
        database.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        await asyncio.gather(
            run_flask(app),
            run_scheduler(
                settings, database, twitter_fetcher, youtube_fetcher,
                generator, verifier, publisher,
            ),
        )
    except Exception as exc:
        logger.error("Main loop error: %s", exc)
        raise
    finally:
        database.close()


if __name__ == "__main__":
    asyncio.run(main())
