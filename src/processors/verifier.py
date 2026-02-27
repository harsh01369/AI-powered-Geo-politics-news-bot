from __future__ import annotations

import logging

from src.models.types import VerificationResult

logger = logging.getLogger(__name__)

CREDIBLE_SOURCES = [
    "bbc.com",
    "reuters.com",
    "apnews.com",
    "aljazeera.com",
    "nytimes.com",
    "thehindu.com",
]


class FactVerifier:
    """Very simple web-scraping fact checker.

    Searches Google for the claim text and looks for links to credible
    news domains.
    """

    def verify(self, claim_text: str) -> VerificationResult:
        try:
            import requests
            from bs4 import BeautifulSoup

            search_url = (
                f"https://www.google.com/search?q="
                f"{requests.utils.quote(claim_text[:100])}"
            )
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36"
                )
            }
            response = requests.get(search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")

            sources_found: list[str] = []
            for link in soup.find_all("a"):
                href = link.get("href", "")
                for src in CREDIBLE_SOURCES:
                    if src in href and href not in sources_found:
                        sources_found.append(href)

            confidence = (
                min(len(sources_found) * 0.4, 0.8)
                if sources_found
                else 0.2
            )
            return VerificationResult(
                verified=confidence >= 0.6,
                sources=sources_found[:2],
                confidence=confidence,
            )
        except Exception as e:
            logger.error("Error verifying claim: %s", e)
            return VerificationResult(
                verified=False, sources=[], confidence=0.0
            )
