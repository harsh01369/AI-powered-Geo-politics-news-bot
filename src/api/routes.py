from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify, request

from src.models.types import Claim, ContentItem, Thread

logger = logging.getLogger(__name__)

api = Blueprint("api", __name__)


def _get_database():
    return current_app.config["DATABASE"]


def _get_publisher():
    return current_app.config["PUBLISHER"]


def _get_generator():
    return current_app.config["GENERATOR"]


def _get_verifier():
    return current_app.config["VERIFIER"]


def _get_settings():
    return current_app.config["SETTINGS"]


def _fetch_url_content(url: str, domain: str) -> ContentItem | None:
    try:
        if "youtube.com" in url or "youtu.be" in url:
            video_id_match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
            if not video_id_match:
                return None
            video_id = video_id_match.group(1)
            return ContentItem(
                id=video_id,
                title=f"YouTube Video {video_id}",
                content="",
                url=url,
                timestamp=datetime.now(tz=timezone.utc),
                source="YouTube",
                language="en",
                domain=domain,
                content_type="youtube",
            )

        import requests
        from bs4 import BeautifulSoup

        response = requests.get(
            url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")[:20]
        text = " ".join(p.get_text() for p in paragraphs if p.get_text())
        text = re.sub(r"\s+", " ", text).strip()[:10_000]
        title = soup.title.string if soup.title else url.split("/")[-1]

        return ContentItem(
            id=str(uuid.uuid4()),
            title=title or "Untitled",
            content=text,
            url=url,
            timestamp=datetime.now(tz=timezone.utc),
            source="Custom URL",
            language="en",
            domain=domain,
            content_type="custom",
        )
    except Exception as exc:
        logger.exception("Failed to fetch URL %s: %s", url, exc)
        return None


@api.route("/generate_thread", methods=["POST"])
def generate_thread() -> tuple:
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    url = data.get("url")
    domain = data.get("domain", "geopolitics")

    if not url:
        return jsonify({"error": "URL is required"}), 400

    settings = _get_settings()
    if domain not in settings.domains:
        valid = list(settings.domains.keys())
        return jsonify({"error": f"Invalid domain. Choose from {valid}"}), 400

    try:
        item = _fetch_url_content(url, domain)
        if item is None:
            return jsonify({"error": "Failed to fetch content from URL"}), 500

        generator = _get_generator()
        thread: Thread = generator.generate(item, domain)

        verifier = _get_verifier()
        claim_text = (item.content or item.title)[:500]
        result = verifier.verify(claim_text)

        db = _get_database()
        db.save_thread(url, domain, item.content, thread)

        claim_id = str(uuid.uuid4())
        claim = Claim(
            id=claim_id,
            source=item.source,
            claim_text=claim_text,
            status="pending",
            timestamp=datetime.now(tz=timezone.utc),
            content_type=f"{domain}_thread",
            post_content="\n".join(thread.posts),
            confidence=result.confidence,
        )
        db.save_claim(claim)
        db.log_metric("threads_generated", 1)

        thread_json = {
            "posts": thread.posts,
            "platform": thread.platform,
            "domain": thread.domain,
        }
        return jsonify({
            "thread": thread.posts,
            "thread_json": thread_json,
            "claim_id": claim_id,
        })
    except Exception as exc:
        logger.exception("Error in generate_thread: %s", exc)
        return jsonify({"error": str(exc)}), 500


@api.route("/claims", methods=["GET"])
def get_claims() -> tuple:
    try:
        db = _get_database()
        claims = db.get_pending_claims()
        for claim in claims:
            if isinstance(claim.get("post_content"), str):
                claim["post_content"] = claim["post_content"].split("\n")
        return jsonify({"claims": claims})
    except Exception as exc:
        logger.exception("Error fetching claims: %s", exc)
        return jsonify({"error": str(exc)}), 500


@api.route("/approve_claim/<claim_id>", methods=["POST"])
def approve_claim(claim_id: str) -> tuple:
    try:
        db = _get_database()
        claim_data = db.get_claim_by_id(claim_id)
        if claim_data is None or claim_data["status"] != "pending":
            return jsonify({"error": "Claim not found or not pending"}), 404

        db.approve_claim(claim_id)

        post_content = claim_data["post_content"]
        posts = post_content.split("\n") if isinstance(post_content, str) else post_content

        publisher = _get_publisher()
        thread = Thread(posts=[p for p in posts if p.strip()], platform="x")
        tweet_ids = publisher.post_thread(thread)

        db.log_metric("posts_approved", 1)
        return jsonify({"status": "approved", "claim_id": claim_id, "tweet_ids": tweet_ids})
    except Exception as exc:
        logger.exception("Error approving claim %s: %s", claim_id, exc)
        return jsonify({"error": str(exc)}), 500


@api.route("/reject_claim/<claim_id>", methods=["POST"])
def reject_claim(claim_id: str) -> tuple:
    try:
        db = _get_database()
        success = db.reject_claim(claim_id)
        if not success:
            return jsonify({"error": "Claim not found or not pending"}), 404

        db.log_metric("posts_rejected", 1)
        return jsonify({"status": "rejected", "claim_id": claim_id})
    except Exception as exc:
        logger.exception("Error rejecting claim %s: %s", claim_id, exc)
        return jsonify({"error": str(exc)}), 500


@api.route("/logs", methods=["GET"])
def get_logs() -> tuple:
    try:
        with open("newsbot.log", "r", encoding="utf-8") as f:
            lines = f.readlines()
        return jsonify({"logs": lines[-50:]})
    except FileNotFoundError:
        return jsonify({"logs": []})
    except Exception as exc:
        logger.exception("Error reading logs: %s", exc)
        return jsonify({"error": str(exc)}), 500
