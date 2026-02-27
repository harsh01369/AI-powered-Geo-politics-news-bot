"""Tests for src.fetchers.utils -- is_relevant, is_sensitive, detect_and_translate.

spaCy and VADER are mocked so that the test suite does not require heavy NLP
models to be installed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# is_relevant
# ---------------------------------------------------------------------------

class TestIsRelevant:
    @patch("src.fetchers.utils.spacy", create=True)
    def test_is_relevant_with_matching_keywords(self, mock_spacy_module):
        """Returns True when at least one token matches a domain/breaking keyword."""
        # Build a fake spaCy pipeline that returns tokens whose .text are
        # the whitespace-split, lowercased words.
        mock_nlp = MagicMock()
        mock_token_diplomacy = MagicMock()
        mock_token_diplomacy.text = "diplomacy"
        mock_token_talks = MagicMock()
        mock_token_talks.text = "talks"
        mock_token_begin = MagicMock()
        mock_token_begin.text = "begin"
        mock_nlp.return_value = [mock_token_diplomacy, mock_token_talks, mock_token_begin]

        with patch("builtins.__import__", side_effect=_make_import_side_effect(mock_nlp)):
            # Re-import so the patched import takes effect inside the function
            from src.fetchers.utils import is_relevant

            result = is_relevant(
                text="Diplomacy talks begin",
                domain_keywords=["diplomacy", "sanctions", "war"],
                breaking_keywords=["breaking", "urgent"],
            )
        assert result is True

    @patch("src.fetchers.utils.spacy", create=True)
    def test_is_relevant_with_no_match(self, mock_spacy_module):
        """Returns False when no token matches any keyword."""
        mock_nlp = MagicMock()
        mock_token_cat = MagicMock()
        mock_token_cat.text = "cat"
        mock_token_video = MagicMock()
        mock_token_video.text = "video"
        mock_nlp.return_value = [mock_token_cat, mock_token_video]

        with patch("builtins.__import__", side_effect=_make_import_side_effect(mock_nlp)):
            from src.fetchers.utils import is_relevant

            result = is_relevant(
                text="Cat video",
                domain_keywords=["diplomacy", "sanctions"],
                breaking_keywords=["breaking"],
            )
        assert result is False


# ---------------------------------------------------------------------------
# is_sensitive
# ---------------------------------------------------------------------------

class TestIsSensitive:
    @patch("src.fetchers.utils.SentimentIntensityAnalyzer", create=True)
    def test_is_sensitive_with_sensitive_content(self, _mock_vader_cls):
        """Returns True when text contains a sensitive keyword AND neg > 0.3."""
        mock_analyzer = MagicMock()
        mock_analyzer.polarity_scores.return_value = {
            "neg": 0.6,
            "neu": 0.3,
            "pos": 0.1,
            "compound": -0.7,
        }
        mock_vader_cls = MagicMock(return_value=mock_analyzer)

        with patch(
            "vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer",
            mock_vader_cls,
        ):
            from src.fetchers.utils import is_sensitive

            result = is_sensitive(
                text="Extreme violence erupted in the region",
                sensitive_keywords=["violence", "terrorism", "hate"],
            )
        assert result is True

    @patch("src.fetchers.utils.SentimentIntensityAnalyzer", create=True)
    def test_is_sensitive_with_safe_content(self, _mock_vader_cls):
        """Returns False when text has no sensitive keywords."""
        mock_analyzer = MagicMock()
        mock_analyzer.polarity_scores.return_value = {
            "neg": 0.05,
            "neu": 0.8,
            "pos": 0.15,
            "compound": 0.3,
        }
        mock_vader_cls = MagicMock(return_value=mock_analyzer)

        with patch(
            "vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer",
            mock_vader_cls,
        ):
            from src.fetchers.utils import is_sensitive

            result = is_sensitive(
                text="The summit concluded with a trade agreement",
                sensitive_keywords=["violence", "terrorism", "hate"],
            )
        assert result is False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_import_side_effect(mock_nlp):
    """Return a side-effect function for patching ``builtins.__import__``
    that intercepts ``import spacy`` and returns a mock whose ``.load()``
    yields *mock_nlp*.
    """
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _side_effect(name, *args, **kwargs):
        if name == "spacy":
            mod = MagicMock()
            mod.load.return_value = mock_nlp
            return mod
        return real_import(name, *args, **kwargs)

    return _side_effect
