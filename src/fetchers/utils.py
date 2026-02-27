from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def detect_and_translate(
    text: str, supported_langs: list[str] | None = None
) -> tuple[str, str]:
    """Detect the language of *text* and translate to English when the detected
    language is neither English nor Hindi and is not in *supported_langs*.

    Returns ``(possibly_translated_text, language_code)``.
    """
    if supported_langs is None:
        supported_langs = ["en", "hi"]

    if not text or not text.strip():
        return text, "en"

    try:
        from langdetect import detect

        lang = detect(text)
    except Exception as exc:
        logger.warning("Language detection failed: %s", exc)
        return text, "en"

    if lang in ("en", "hi"):
        return text, lang

    try:
        from deep_translator import GoogleTranslator

        translated = GoogleTranslator(source=lang, target="en").translate(
            text[:5000]
        )
        return translated, "en"
    except Exception as exc:
        logger.warning("Translation from '%s' failed: %s", lang, exc)
        return text, lang


def is_relevant(
    text: str,
    domain_keywords: list[str],
    breaking_keywords: list[str],
) -> bool:
    """Return ``True`` if *text* contains at least one domain keyword or one
    breaking keyword, checked via spaCy tokenisation."""
    if not text:
        return False

    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text.lower())
        all_keywords = set(domain_keywords) | set(breaking_keywords)
        return any(token.text in all_keywords for token in doc)
    except Exception as exc:
        logger.warning("Relevance check failed: %s", exc)
        return False


def is_sensitive(text: str, sensitive_keywords: list[str]) -> bool:
    """Return ``True`` when *text* contains at least one sensitive keyword
    **and** VADER detects negative sentiment above 0.3."""
    if not text:
        return False

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        text_lower = text.lower()
        has_keyword = any(kw in text_lower for kw in sensitive_keywords)
        if not has_keyword:
            return False

        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        return scores["neg"] > 0.3
    except Exception as exc:
        logger.warning("Sensitivity check failed: %s", exc)
        return True
