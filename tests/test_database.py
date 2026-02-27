"""Tests for src.storage.database.Database."""
from __future__ import annotations

from datetime import datetime

from src.models.types import Claim


class TestSaveAndRetrieveClaim:
    def test_save_and_retrieve_claim(self, temp_database):
        claim = Claim(
            id="claim-100",
            source="Reuters",
            claim_text="A major summit was held today.",
            status="pending",
            timestamp=datetime(2025, 6, 15, 10, 0, 0),
            content_type="news",
            post_content="Breaking: summit held today",
            confidence=0.75,
        )
        temp_database.save_claim(claim)

        pending = temp_database.get_pending_claims()
        assert len(pending) == 1
        assert pending[0]["id"] == "claim-100"
        assert pending[0]["source"] == "Reuters"
        assert pending[0]["claim"] == "A major summit was held today."
        assert pending[0]["content_type"] == "news"
        assert pending[0]["post_content"] == "Breaking: summit held today"
        assert pending[0]["confidence"] == 0.75


class TestApproveClaim:
    def test_approve_claim(self, temp_database):
        claim = Claim(
            id="claim-200",
            source="BBC",
            claim_text="Elections announced.",
            status="pending",
            timestamp=datetime(2025, 7, 1, 8, 0, 0),
            content_type="news",
            post_content="Breaking: elections announced",
            confidence=0.80,
        )
        temp_database.save_claim(claim)

        result = temp_database.approve_claim("claim-200")
        assert result is True

        # After approval the claim should no longer appear in pending list
        pending = temp_database.get_pending_claims()
        assert len(pending) == 0

        # Verify via get_claim_by_id that the status actually changed
        row = temp_database.get_claim_by_id("claim-200")
        assert row is not None
        assert row["status"] == "approved"


class TestRejectClaim:
    def test_reject_claim(self, temp_database):
        claim = Claim(
            id="claim-300",
            source="AP",
            claim_text="Trade deal signed.",
            status="pending",
            timestamp=datetime(2025, 8, 20, 14, 0, 0),
            content_type="news",
            post_content="Breaking: trade deal signed",
            confidence=0.65,
        )
        temp_database.save_claim(claim)

        result = temp_database.reject_claim("claim-300")
        assert result is True

        pending = temp_database.get_pending_claims()
        assert len(pending) == 0

        row = temp_database.get_claim_by_id("claim-300")
        assert row is not None
        assert row["status"] == "rejected"


class TestIsProcessed:
    def test_is_processed(self, temp_database):
        assert temp_database.is_processed("content-abc") is False
        temp_database.mark_processed("content-abc", source="Reuters")
        assert temp_database.is_processed("content-abc") is True


class TestDuplicateDetection:
    def test_duplicate_detection(self, temp_database):
        """mark_processed then is_processed returns True for the same content_id."""
        content_id = "dup-content-001"
        temp_database.mark_processed(content_id, source="BBC")
        assert temp_database.is_processed(content_id) is True

        # A different content_id should not be detected as processed
        assert temp_database.is_processed("other-content-999") is False
