![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Twitter API](https://img.shields.io/badge/Twitter%20API-v2-1DA1F2?logo=twitter&logoColor=white)

# AI-Powered Geopolitics News Bot

An end-to-end AI automation system that fetches, analyzes, verifies, and distributes geopolitical news. It pulls content from multiple sources, processes 12+ languages, scores relevance, verifies facts, and posts approved stories to social media.

## How It Works

The system runs a seven-stage pipeline. Raw content enters at one end. Verified, summarized posts come out the other.

### Stage 1: Data Collection

Three source types run concurrently, each in its own thread:

- **YouTube** — searches geopolitical keywords, extracts video metadata and transcripts via the YouTube Data API v3.
- **RSS Feeds** — monitors international news sources using feed parsing.
- **Twitter/X** — tracks accounts and hashtags related to geopolitics via the Twitter API v2.

### Stage 2: Language Detection and Translation

- Detects content language from 12+ supported languages.
- Translates non-English content to English for downstream processing.
- Preserves original language metadata for reference.

### Stage 3: NLP Filtering

- SpaCy extracts named entities, keywords, and topics from each article.
- VADER sentiment analysis scores the emotional tone.
- A relevance score is computed based on keyword frequency and entity matching.
- Low-relevance content is filtered out before reaching the Transformer model.

### Stage 4: AI Summarization

- The BART Transformer model (via Hugging Face) generates concise summaries.
- Key facts are preserved while content length is reduced.
- Output is formatted as social-media-ready text.

### Stage 5: Fact Verification

- Claims are cross-referenced across multiple sources.
- Each story receives a confidence score from 0 to 100.
- Single-source or unverified stories are flagged.

### Stage 6: Human Moderation

- Verified stories enter a moderation queue.
- A human reviewer approves or rejects each story before publishing.
- This prevents false or misleading content from being posted.

### Stage 7: Social Media Distribution

- Approved content is posted to Twitter/X.
- Text is formatted for platform character limits.
- Source attribution is included.

## Pipeline Diagram

```
YouTube ──┐
RSS ──────┤──▶ Language Detection ──▶ NLP Filtering ──▶ AI Summarization
Twitter ──┘         & Translation       (SpaCy/VADER)     (BART Model)
                                                              │
                                                              ▼
                                Twitter/X ◀── Human Moderation ◀── Fact Verification
                                 Posting        Queue              (Confidence Score)
```

## Features

- Multi-source data collection (YouTube, RSS, Twitter/X)
- 12+ language support with automatic translation
- NLP-based relevance filtering using SpaCy and VADER
- AI summarization using BART Transformers
- Fact verification with confidence scoring (0-100)
- Human moderation queue with approve/reject workflow
- Automated social media posting to Twitter/X
- Real-time log monitoring
- Multi-threaded concurrent processing
- SQLite storage for articles and processing state

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Web API | Flask |
| Deep Learning | TensorFlow |
| Summarization | Hugging Face Transformers (BART) |
| NLP | SpaCy |
| Sentiment Analysis | VADER |
| Social Media | Twitter API v2 |
| Video Data | YouTube Data API v3 |
| News Feeds | RSS feed parsing |
| Database | SQLite |
| Web Scraping | BeautifulSoup |
| Concurrency | Multi-threaded processing |

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Twitter API v2 credentials (API key, API secret, access token, access token secret)
- YouTube Data API v3 key

### Installation

1. Clone the repository:

```bash
git clone https://github.com/harsh01369/AI-powered-Geo-politics-news-bot.git
cd AI-powered-Geo-politics-news-bot
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set environment variables:

```bash
export TWITTER_API_KEY="your_api_key"
export TWITTER_API_SECRET="your_api_secret"
export TWITTER_ACCESS_TOKEN="your_access_token"
export TWITTER_ACCESS_SECRET="your_access_token_secret"
export YOUTUBE_API_KEY="your_youtube_api_key"
```

4. Start the Flask server:

```bash
python app.py
```

5. Open the moderation dashboard at `http://localhost:5000`.

## Design Decisions

**Multi-threaded collection.** Each source (YouTube, RSS, Twitter) runs in its own thread. This reduces total fetch time because sources are polled concurrently instead of sequentially.

**NLP pre-filtering.** SpaCy and VADER process articles before the Transformer model sees them. This prevents expensive BART inference on irrelevant content, saving compute time and cost.

**Confidence scoring.** Each story gets a score from 0 to 100 based on cross-source verification. This is more useful than a binary true/false label. It lets the human reviewer make informed decisions based on the strength of evidence.

**Human-in-the-loop.** The AI processes and scores content. A human approves it before publishing. This prevents automated misinformation. No story reaches social media without human review.

**SQLite.** A zero-config database that requires no separate server. It is sufficient for storing articles, processing state, and moderation queue data in a single-instance pipeline.

## License

This project is open source. See the [LICENSE](LICENSE) file for details.
