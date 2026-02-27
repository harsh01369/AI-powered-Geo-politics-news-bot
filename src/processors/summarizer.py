from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class Summarizer:
    """Wraps a HuggingFace summarisation pipeline.

    The heavy model is loaded lazily on first call so that construction is
    cheap and tests can easily mock ``self._pipeline``.
    """

    def __init__(self) -> None:
        self._pipeline = None

    def _load_pipeline(self) -> None:
        try:
            from transformers import pipeline as hf_pipeline

            self._pipeline = hf_pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                framework="pt",
                device=-1,
            )
            logger.info("Initialised summarisation pipeline")
        except Exception as e:
            logger.warning("Failed to initialise summariser: %s", e)
            self._pipeline = False  # sentinel: don't retry

    def summarize(self, content: str) -> str:
        """Return a short summary of *content*."""
        if self._pipeline is None:
            self._load_pipeline()

        if self._pipeline is False:
            return content[:100]

        try:
            content = content[:512]
            word_count = len(content.split())
            max_len = min(100, word_count // 2 + 10) if word_count > 20 else 10
            result = self._pipeline(
                content, max_length=max_len, min_length=10, do_sample=False
            )
            return result[0]["summary_text"]
        except Exception as e:
            logger.error("Error summarising content: %s", e)
            return content[:100]

    def extract_key_points(self, content: str, num_points: int = 5) -> list[str]:
        """Extract up to *num_points* key points from *content*."""
        try:
            from nltk.tokenize import sent_tokenize

            sentences = sent_tokenize(content)[:num_points]
            return sentences if sentences else [content[:100]]
        except Exception as e:
            logger.error("Error extracting key points: %s", e)
            return [content[:100]]
