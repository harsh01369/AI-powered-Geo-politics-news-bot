from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid

from src.models.types import Claim, Thread

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: str = "news_bot.db") -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS claims (
                    id TEXT PRIMARY KEY,
                    source TEXT,
                    claim TEXT,
                    status TEXT,
                    timestamp TEXT,
                    content_type TEXT,
                    post_content TEXT,
                    confidence REAL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_content (
                    id TEXT PRIMARY KEY,
                    source TEXT,
                    content_id TEXT,
                    timestamp TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    id TEXT PRIMARY KEY,
                    metric_name TEXT,
                    value INTEGER,
                    timestamp TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    url TEXT,
                    domain TEXT,
                    content TEXT,
                    thread_json TEXT,
                    timestamp TEXT
                )
                """
            )
            self._conn.commit()

    def save_claim(self, claim: Claim) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO claims "
                "(id, source, claim, status, timestamp, content_type, post_content, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    claim.id,
                    claim.source,
                    claim.claim_text,
                    claim.status,
                    claim.timestamp.isoformat()
                    if hasattr(claim.timestamp, "isoformat")
                    else str(claim.timestamp),
                    claim.content_type,
                    claim.post_content,
                    claim.confidence,
                ),
            )
            self._conn.commit()

    def get_pending_claims(self) -> list[dict]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id, source, claim, content_type, post_content, confidence "
                "FROM claims WHERE status = 'pending'"
            )
            rows = cursor.fetchall()
        return [
            {
                "id": row["id"],
                "source": row["source"],
                "claim": row["claim"],
                "content_type": row["content_type"],
                "post_content": row["post_content"],
                "confidence": row["confidence"],
            }
            for row in rows
        ]

    def approve_claim(self, claim_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE claims SET status = 'approved' WHERE id = ? AND status = 'pending'",
                (claim_id,),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def reject_claim(self, claim_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE claims SET status = 'rejected' WHERE id = ? AND status = 'pending'",
                (claim_id,),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def is_processed(self, content_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT 1 FROM processed_content WHERE content_id = ?",
                (content_id,),
            )
            return cursor.fetchone() is not None

    def mark_processed(self, content_id: str, source: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO processed_content (id, source, content_id, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    source,
                    content_id,
                    time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                ),
            )
            self._conn.commit()

    def save_thread(self, url: str, domain: str, content: str, thread: Thread) -> str:
        thread_id = str(uuid.uuid4())
        thread_json = json.dumps(
            {"posts": thread.posts, "platform": thread.platform, "domain": thread.domain}
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO threads (id, url, domain, content, thread_json, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    thread_id,
                    url,
                    domain,
                    content[:1000],
                    thread_json,
                    time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                ),
            )
            self._conn.commit()
        return thread_id

    def log_metric(self, metric_name: str, value: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO metrics (id, metric_name, value, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    metric_name,
                    value,
                    time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                ),
            )
            self._conn.commit()

    def get_claim_by_id(self, claim_id: str) -> dict | None:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id, source, claim, status, post_content, content_type, confidence "
                "FROM claims WHERE id = ?",
                (claim_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "source": row["source"],
            "claim": row["claim"],
            "status": row["status"],
            "post_content": row["post_content"],
            "content_type": row["content_type"],
            "confidence": row["confidence"],
        }

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
                self._conn.close()
                logger.info("Database connection closed")
            except Exception as exc:
                logger.error("Error closing database: %s", exc)
