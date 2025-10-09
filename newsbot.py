import os
import sys
import json
import tweepy
import sqlite3
import uuid
import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
import spacy
import schedule
import time
import asyncio
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import re
import feedparser
import nltk
from nltk.tokenize import sent_tokenize
from langdetect import detect, DetectorFactory
from concurrent.futures import ThreadPoolExecutor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import threading
import signal
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import RequestException
from deep_translator import GoogleTranslator
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import tensorflow as tf

# Ensure deterministic language detection
DetectorFactory.seed = 0

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Set up robust logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

log_file = 'newsbot.log'
try:
    if os.path.exists(log_file) and not os.access(log_file, os.W_OK):
        print(f"Warning: No write permission for {log_file}. Logging to stderr.", file=sys.stderr)
        file_handler = None
    else:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
except Exception as e:
    print(f"Error setting up file logging: {e}. Logging to stderr.", file=sys.stderr)
    file_handler = None

stream_handler = logging.StreamHandler(sys.stderr if sys.platform == 'win32' else sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.WARNING)

# Load environment variables
load_dotenv()
X_API_KEY = os.getenv("X_API_KEY")
X_API_SECRET = os.getenv("X_API_SECRET")
X_ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN")
X_ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_TOKEN_SECRET")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Load configuration
CONFIG_FILE = 'newsbot_config.json'
DEFAULT_CONFIG = {
    "x_accounts": ["ArmMonitor11"],
    "youtube_channels": [
        "UCsT0YIqwnpJCM-mx7-gSA4Q",
        "UCt-ybO9Kw9QqG9Ts_YaPJgA",
        "UC3cU0KXMBOKh3gK-2U8iHqw",
        "UCSYMy1wJ0gtM3HLoC4SnuRw",
        "UCrC8mOqJQpoB7NuIMKIS6rQ",
        "UC2rGfsVex4dgKJBvFzUh-Lg",
        "UCSiDGb0MnHFGjs5UbhZ1Qaw"
    ],
    "rss_feeds": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://feeds.reuters.com/reuters/topNews",
        "https://www.aljazeera.com/xml/rss/all.xml",
        "https://www.thehindu.com/news/international/feeder/default.rss",
        "https://apnews.com/hub/world-news/rss"
    ],
    "sensitive_keywords": ["violence", "hate", "graphic", "terrorism", "explicit", "riot", "radicalization", "communal"],
    "relevance_keywords": ["politics", "geopolitics", "diplomacy", "policy", "government", "trade", "conflict", "summit", "election", "sanction", "war", "treaty", "alliance"],
    "breaking_keywords": ["breaking", "urgent", "summit", "conflict", "crisis", "announce", "declare"],
    "schedule_interval_minutes": 15,
    "supported_languages": ["en", "hi", "es", "fr", "de", "ar", "ru", "pt", "zh-Hans", "zh-Hant", "it", "ja"],
    "test_mode": True,
    "domains": {
        "geopolitics": {
            "keywords": ["politics", "geopolitics", "diplomacy", "conflict", "war", "treaty"],
            "context": {
                "israel_iran": [
                    "The Israel-Iran conflict intensified after the 1979 Iranian Revolution, which brought an anti-US/Israel regime under Khomeini to power.",
                    "The US 'Atoms for Peace' program in the 1950s provided Iran with nuclear technology, contributing to its current capabilities.",
                    "Israel and the US have supported groups like MEK to destabilize Iran, including during the 2009-2010 Green Movement."
                ],
                "nato": [
                    "NATO, founded in 1949, aims to ensure collective defense among member states, often influencing global geopolitics.",
                    "Recent NATO summits focus on countering Russia and China's growing influence."
                ],
                "us_china": [
                    "US-China tensions escalated post-2018 due to trade wars and technological competition.",
                    "China strengthens ties with Russia and Iran for energy security."
                ],
                "india_pakistan": [
                    "India-Pakistan tensions stem from the 1947 partition, with conflicts over Kashmir.",
                    "Recent diplomatic efforts focus on trade and counter-terrorism."
                ],
                "russia_ukraine": [
                    "Russia-Ukraine tensions escalated after the 2014 annexation of Crimea.",
                    "The 2022 invasion intensified global sanctions and NATO involvement."
                ]
            }
        },
        "spiritual": {
            "keywords": ["spirituality", "meditation", "yoga", "enlightenment", "philosophy"],
            "context": {
                "indian_spirituality": [
                    "Indian spiritual traditions like Advaita Vedanta emphasize non-duality and self-realization.",
                    "Yoga and meditation practices trace back to texts like the Bhagavad Gita."
                ]
            }
        }
    }
}
try:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
except Exception as e:
    logger.error(f"Failed to load config: {e}. Using default config.")
    config = DEFAULT_CONFIG

# Validate environment variables
if not all([X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET, YOUTUBE_API_KEY]):
    logger.error("Missing environment variables. Please check .env file.")
    sys.exit(1)

# NLP and sentiment tools
try:
    nlp = spacy.load("en_core_web_sm")
    nlp_hi = spacy.load("xx_ent_wiki_sm")
    sentiment_analyzer = SentimentIntensityAnalyzer()
except Exception as e:
    logger.error(f"Failed to initialize NLP tools: {e}")
    sys.exit(1)

# Initialize summarizer and key point extractor
summarizer = None
keypoint_model = None
def init_nlp_models():
    global summarizer, keypoint_model
    if summarizer is None:
        try:
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt", device=-1)
            logger.info("Initialized summarizer")
        except Exception as e:
            logger.warning(f"Failed to initialize summarizer: {e}. Using raw text fallback.")
            summarizer = False
    if keypoint_model is None:
        try:
            model_name = "facebook/bart-large"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            keypoint_model = pipeline("summarization", model=model, tokenizer=tokenizer, framework="pt", device=-1)
            logger.info("Initialized keypoint extractor")
        except Exception as e:
            logger.warning(f"Failed to initialize keypoint extractor: {e}. Using sentence splitting fallback.")
            keypoint_model = False

# Initialize X API
try:
    client = tweepy.Client(
        consumer_key=X_API_KEY,
        consumer_secret=X_API_SECRET,
        access_token=X_ACCESS_TOKEN,
        access_token_secret=X_ACCESS_TOKEN_SECRET
    )
    user = client.get_me().data
    logger.info(f"X API initialized for user {user.username}")
except Exception as e:
    logger.error(f"Failed to initialize X API: {e}. Check credentials or API tier.")
    sys.exit(1)

# Initialize YouTube API
try:
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    youtube.videos().list(part="snippet", id="RY9HFhHYrZQ").execute()
except Exception as e:
    logger.error(f"Failed to initialize YouTube API: {e}. Check API key or video access.")
    sys.exit(1)

# Flask app for UI integration
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:8000"}})

# Thread-local storage for SQLite connections
thread_local = threading.local()
db_lock = threading.Lock()

# Global event loop
loop = asyncio.get_event_loop()

def get_db_connection():
    if not hasattr(thread_local, "conn"):
        try:
            thread_local.conn = sqlite3.connect("news_bot.db", check_same_thread=False)
            thread_local.cursor = thread_local.conn.cursor()
            with db_lock:
                thread_local.cursor.execute("""
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
                """)
                thread_local.cursor.execute("""
                    CREATE TABLE IF NOT EXISTS processed_content (
                        id TEXT PRIMARY KEY,
                        source TEXT,
                        content_id TEXT,
                        timestamp TEXT
                    )
                """)
                thread_local.cursor.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        id TEXT PRIMARY KEY,
                        metric_name TEXT,
                        value INTEGER,
                        timestamp TEXT
                    )
                """)
                thread_local.cursor.execute("""
                    CREATE TABLE IF NOT EXISTS threads (
                        id TEXT PRIMARY KEY,
                        url TEXT,
                        domain TEXT,
                        content TEXT,
                        thread_json TEXT,
                        timestamp TEXT
                    )
                """)
                thread_local.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
            sys.exit(1)
    return thread_local.conn, thread_local.cursor

def close_db_connection():
    if hasattr(thread_local, "conn"):
        with db_lock:
            try:
                thread_local.conn.commit()
                thread_local.conn.close()
                del thread_local.conn
                del thread_local.cursor
                logger.info("Closed thread-local SQLite connection")
            except Exception as e:
                logger.error(f"Error closing SQLite connection: {e}")

def log_metric(metric_name, value):
    try:
        conn, cursor = get_db_connection()
        cursor.execute(
            "INSERT INTO metrics (id, metric_name, value, timestamp) VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), metric_name, value, time.strftime("%Y-%m-%dT%H:%M:%SZ"))
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Error logging metric {metric_name}: {e}")

def is_relevant(text, domain):
    try:
        doc = nlp(text.lower())
        return any(token.text in config["domains"][domain]["keywords"] or token.text in config["breaking_keywords"] for token in doc)
    except Exception as e:
        logger.warning(f"Error in is_relevant: {e}")
        return False

def is_sensitive(text):
    try:
        text_lower = text.lower()
        keyword_match = any(keyword in text_lower for keyword in config["sensitive_keywords"])
        sentiment = sentiment_analyzer.polarity_scores(text)
        return keyword_match and sentiment["neg"] > 0.3
    except Exception as e:
        logger.warning(f"Error in is_sensitive: {e}")
        return True

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(RequestException))
def fetch_x_posts():
    if config["test_mode"]:
        logger.info("Test mode: Returning mock X posts")
        return [{
            "id": "mock_1",
            "account": "ArmMonitor11",
            "text": "Breaking: Summit on Israel-Iran tensions. #Geopolitics",
            "url": "https://x.com/ArmMonitor11/status/mock_1",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }]
    posts = []
    try:
        logger.info(f"Fetching posts for {user.username}")
        tweets = client.get_users_tweets(
            id=user.id,
            tweet_fields=["created_at", "text"],
            max_results=5
        )
        if tweets.data:
            for tweet in tweets.data:
                if is_relevant(tweet.text, "geopolitics") and not is_sensitive(tweet.text):
                    posts.append({
                        "id": str(tweet.id),
                        "account": user.username,
                        "text": tweet.text,
                        "url": f"https://x.com/{user.username}/status/{tweet.id}",
                        "timestamp": str(tweet.created_at)
                    })
                else:
                    logger.info(f"Skipped tweet {tweet.id}: not relevant or sensitive")
            log_metric("x_posts_fetched", len(tweets.data))
        logger.info(f"Successfully fetched {len(tweets.data or [])} posts for {user.username}")
    except tweepy.TweepyException as e:
        logger.error(f"Error fetching posts: {e}")
        if e.response and e.response.status_code == 401:
            logger.error("401 Unauthorized. Verify X API credentials.")
        elif e.response and e.response.status_code == 429:
            logger.error("Rate limit exceeded. Waiting 15 minutes.")
            time.sleep(900)
        return []
    return posts

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(RequestException))
def fetch_youtube_videos():
    if config["test_mode"]:
        logger.info("Test mode: Returning mock YouTube videos")
        return [{
            "id": "RY9HFhHYrZQ",
            "title": "Breaking: Geopolitical Summit Analysis",
            "description": "Analysis of Israel-Iran relations.",
            "transcript": "Detailed discussion on recent summit outcomes and Israel-Iran tensions.",
            "transcript_data": [{"text": "Summit discusses peace talks.", "start": 0, "duration": 5}],
            "url": "https://www.youtube.com/watch?v=RY9HFhHYrZQ",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "Test Source",
            "language": "en"
        }]
    videos = []
    transcript_warnings = set()
    for channel_id in config["youtube_channels"]:
        try:
            logger.info(f"Fetching videos for channel {channel_id}")
            request = youtube.search().list(
                part="snippet",
                channelId=channel_id,
                maxResults=1,  # One video to conserve quota
                order="date"
            )
            response = request.execute()
            for item in response.get("items", []):
                video_id = item["id"]["videoId"]
                title = item["snippet"]["title"]
                description = item["snippet"]["description"]
                content = ""
                transcript_data = []
                try:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=config["supported_languages"])
                    transcript_data = transcript
                    content = " ".join([entry["text"] for entry in transcript])  # Prioritize subtitles
                except Exception as e:
                    if channel_id not in transcript_warnings:
                        logger.warning(f"No transcript for channel {channel_id}: {str(e)[:100]}")
                        transcript_warnings.add(channel_id)
                    content = title + " " + description  # Fallback to title+description
                    try:
                        comment_request = youtube.commentThreads().list(
                            part="snippet",
                            videoId=video_id,
                            maxResults=3,
                            textFormat="plainText"
                        )
                        comment_response = comment_request.execute()
                        comments = [item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                                   for item in comment_response.get("items", [])]
                        content += " " + " ".join(comments)
                    except Exception as e:
                        logger.debug(f"No comments for video {video_id}: {str(e)[:100]}")
                try:
                    lang = detect(content)
                    if lang not in ["en", "hi"]:
                        content = GoogleTranslator(source=lang, target="en").translate(content[:5000])
                        lang = "en"
                except Exception:
                    lang = "en"
                    logger.warning(f"Language detection/translation failed for video {video_id}, defaulting to 'en'")
                if is_relevant(content, "geopolitics") and not is_sensitive(content):
                    source = {
                        "UCSiDGb0MnHFGjs5UbhZ1Qaw": "Lex Fridman",
                        "UC2rGfsVex4dgKJBvFzUh-Lg": "BeerBiceps",
                        "UCrC8mOqJQpoB7NuIMKIS6rQ": "Joe Rogan",
                        "UCSYMy1wJ0gtM3HLoC4SnuRw": "Abhijit Chavda",
                        "UC3cU0KXMBOKh3gK-2U8iHqw": "Vikas Divyakirti",
                        "UCt-ybO9Kw9QqG9Ts_YaPJgA": "Nitish Rajput",
                        "UCsT0YIqwnpJCM-mx7-gSA4Q": "Pavneet Singh"
                    }.get(channel_id, "Unknown")
                    videos.append({
                        "id": video_id,
                        "title": title,
                        "description": description,
                        "transcript": content[:10000],
                        "transcript_data": transcript_data,
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                        "timestamp": item["snippet"]["publishedAt"],
                        "source": source,
                        "language": lang
                    })
                else:
                    logger.info(f"Skipped video {video_id}: not relevant or sensitive")
            log_metric("youtube_videos_fetched", len(response.get('items', [])))
        except Exception as e:
            logger.error(f"Error fetching videos for channel {channel_id}: {e}")
            if "quotaExceeded" in str(e):
                logger.error("YouTube API quota exceeded. Using test mode fallback.")
                return []
    return videos

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(RequestException))
def fetch_news():
    if config["test_mode"]:
        logger.info("Test mode: Returning mock news items")
        return [{
            "id": str(uuid.uuid4()),
            "title": "Breaking: Geopolitical Summit Announced",
            "content": "A major summit addressing Israel-Iran tensions is set for next week.",
            "url": "https://example.com/news",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "Test News"
        }]
    news = []
    for feed_url in config["rss_feeds"]:
        try:
            logger.info(f"Fetching news from {feed_url}")
            feed = feedparser.parse(feed_url)
            entries = feed.entries[:5]  # Fetch more for breaking news
            for entry in entries:
                title = entry.get("title", "")
                link = entry.get("link", "")
                published = entry.get("published", time.strftime("%Y-%m-%dT%H:%M:%SZ"))
                content = ""
                try:
                    response = requests.get(link, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                    soup = BeautifulSoup(response.text, "html.parser")
                    paragraphs = soup.find_all("p")[:20]
                    content = " ".join(p.get_text() for p in paragraphs if p.get_text())
                    content = re.sub(r'\s+', ' ', content).strip()[:10000]
                except Exception as e:
                    logger.warning(f"Failed to fetch full text for {link}: {e}")
                    content = entry.get("summary", title)[:1000]
                try:
                    lang = detect(content)
                    if lang not in ["en", "hi"]:
                        content = GoogleTranslator(source=lang, target="en").translate(content[:5000])
                        lang = "en"
                except Exception:
                    lang = "en"
                    logger.warning(f"Language detection/translation failed for news {link}, defaulting to 'en'")
                if is_relevant(content, "geopolitics") and not is_sensitive(content):
                    news.append({
                        "id": str(uuid.uuid4()),
                        "title": title,
                        "content": content,
                        "url": link,
                        "timestamp": published,
                        "source": feed.feed.get("title", "Unknown")
                    })
                else:
                    logger.info(f"Skipped news item {title[:50]} from {feed_url}: not relevant or sensitive")
            log_metric("news_items_fetched", len(entries))
        except Exception as e:
            logger.error(f"Error fetching news from {feed_url}: {e}")
    return news

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(RequestException))
def fetch_custom_url(url, domain="geopolitics"):
    if config["test_mode"]:
        logger.info("Test mode: Returning mock custom URL content")
        return {
            "id": "RY9HFhHYrZQ",
            "title": "Breaking: Custom Geopolitical Video",
            "description": "Analysis of recent summit.",
            "transcript": "Detailed discussion on peace talks and conflicts.",
            "transcript_data": [{"text": "Summit addresses key issues.", "start": 0, "duration": 5}],
            "url": url,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "Custom YouTube",
            "language": "en",
            "domain": domain
        }
    try:
        logger.info(f"Fetching custom URL {url} for domain {domain}")
        if "youtube.com" in url or "youtu.be" in url:
            video_id = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
            if not video_id:
                logger.warning(f"Invalid YouTube URL: {url}")
                return None
            video_id = video_id.group(1)
            request = youtube.videos().list(part="snippet", id=video_id)
            response = request.execute()
            if not response.get("items", []):
                logger.warning(f"No video found for ID {video_id}")
                return None
            item = response["items"][0].get("snippet", {})
            title = item.get("title", "Unknown")
            description = item.get("description", "")
            content = ""
            transcript_data = []
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=config["supported_languages"])
                transcript_data = transcript
                content = " ".join([entry["text"] for entry in transcript])  # Prioritize subtitles
            except Exception as e:
                logger.warning(f"No transcript for video {video_id}: {str(e)[:100]}")
                content = title + " " + description
                try:
                    comment_request = youtube.commentThreads().list(
                        part="snippet",
                        videoId=video_id,
                        maxResults=3,
                        textFormat="plainText"
                    )
                    comment_response = comment_request.execute()
                    comments = [item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                               for item in comment_response.get("items", [])]
                    content += " " + " ".join(comments)
                except Exception as e:
                    logger.debug(f"No comments for video {video_id}: {str(e)[:100]}")
            try:
                lang = detect(content)
                if lang not in ["en", "hi"]:
                    content = GoogleTranslator(source=lang, target="en").translate(content[:5000])
                    lang = "en"
            except Exception:
                lang = "en"
                logger.warning(f"Language detection/translation failed for video {video_id}, defaulting to 'en'")
            return {
                "id": video_id,
                "title": title,
                "description": description,
                "transcript": content[:10000],
                "transcript_data": transcript_data,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source": "Custom YouTube",
                "language": lang,
                "domain": domain
            }
        else:
            response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(response.text, "html.parser")
            text = " ".join(p.get_text() for p in soup.find_all("p")[:20])
            text = re.sub(r'\s+', ' ', text).strip()[:10000]
            title = soup.title.string if soup.title else url.split("/")[-1]
            try:
                lang = detect(text)
                if lang not in ["en", "hi"]:
                    text = GoogleTranslator(source=lang, target="en").translate(text[:5000])
                    lang = "en"
            except Exception:
                lang = "en"
                logger.warning(f"Language detection/translation failed for URL {url}, defaulting to 'en'")
            return {
                "id": str(uuid.uuid4()),
                "title": title,
                "content": text,
                "url": url,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source": "Custom URL",
                "language": lang,
                "domain": domain
            }
    except Exception as e:
        logger.error(f"Error fetching custom URL {url}: {e}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(RequestException))
def verify_claim(claim):
    if config["test_mode"]:
        logger.info("Test mode: Returning mock verification")
        return True, ["https://example.com"], 0.8
    try:
        search_url = f"https://www.google.com/search?q={requests.utils.quote(claim[:100])}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        credible_sources = ["bbc.com", "reuters.com", "apnews.com", "aljazeera.com", "nytimes.com", "thehindu.com"]
        sources_found = []
        for link in soup.find_all("a"):
            href = link.get("href", "")
            for source in credible_sources:
                if source in href and href not in sources_found:
                    sources_found.append(href)
        confidence = min(len(sources_found) * 0.4, 0.8) if sources_found else 0.2
        logger.info(f"Claim verified: {confidence:.2f} confidence, sources {sources_found[:2]}")
        return confidence >= 0.6, sources_found[:2], confidence
    except Exception as e:
        logger.error(f"Error verifying claim with Google: {e}")
        return False, [], 0.0

def summarize_content(content):
    if summarizer is None:
        init_nlp_models()
    if summarizer is False:
        return content[:100]
    try:
        content = content[:512]  # Strict limit for BART
        input_length = len(nlp(content).text.split())
        adjusted_max_length = min(100, input_length // 2 + 10) if input_length > 20 else 10
        logger.info(f"Summarizing content with length {input_length}, max_length={adjusted_max_length}")
        summary = summarizer(content, max_length=adjusted_max_length, min_length=10, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        logger.error(f"Error summarizing content: {e}")
        return content[:100]

def extract_key_points(content, num_points=5):
    if keypoint_model is None:
        init_nlp_models()
    if keypoint_model is False:
        sentences = sent_tokenize(content)[:num_points]
        return sentences if sentences else [content[:100]]
    try:
        content = content[:512]  # Strict limit for BART
        sentences = sent_tokenize(content)
        points = []
        chunk_size = max(1, len(sentences) // num_points)
        for i in range(0, len(sentences), chunk_size):
            chunk = " ".join(sentences[i:i+chunk_size])
            if chunk:
                input_length = len(nlp(chunk).text.split())
                adjusted_max_length = min(50, input_length // 2 + 5) if input_length > 10 else 10
                summary = keypoint_model(chunk, max_length=adjusted_max_length, min_length=10, do_sample=False)
                points.append(summary[0]["summary_text"])
            if len(points) >= num_points:
                break
        return points[:num_points] if points else [content[:100]]
    except Exception as e:
        logger.error(f"Error extracting key points: {e}")
        return sent_tokenize(content)[:num_points] if sent_tokenize(content) else [content[:100]]

def generate_rich_thread(item, domain):
    try:
        title = item.get("title", "Untitled")
        source = item.get("source", "Unknown")
        content = item.get("transcript", item.get("content", item.get("description", title)))[:10000]
        content = re.sub(r'(UPI for support|PayPal|https?://\S+)', '', content, flags=re.IGNORECASE)
        summary = summarize_content(content)
        key_points = extract_key_points(content, num_points=5)
        
        thread = [f"Breaking: {title[:80]} 🌍 #{domain.capitalize()} {item.get('url', '')}"[:280]]
        thread.append(f"1/7: {source} reports: {summary[:100]}... #WorldAffairs"[:280])
        
        context_key = None
        title_lower = title.lower()
        content_lower = content.lower()
        for key in config["domains"][domain]["context"]:
            if any(k in title_lower or k in content_lower for k in key.split("_")):
                context_key = key
                break
        
        if context_key:
            for i, ctx in enumerate(config["domains"][domain]["context"][context_key][:2], 2):
                thread.append(f"{i}/7: Context: {ctx[:100]}... #{domain.capitalize()}"[:280])
            offset = 4
            for i, point in enumerate(key_points[:3], offset):
                thread.append(f"{i}/7: Key Point: {point[:100]}... #{domain.capitalize()}"[:280])
        else:
            for i, point in enumerate(key_points[:5], 2):
                thread.append(f"{i}/7: Key Point: {point[:100]}... #{domain.capitalize()}"[:280])
            offset = len(key_points) + 2
        
        thread.append(f"{offset}/7: What's your take on {title[:50]}...? Share below! 🔍 #{domain.capitalize()}"[:280])
        
        if item.get("content_type") == "news":
            thread = [
                f"Breaking: {title[:80]} from {source}. 🌍 #{domain.capitalize()} {item.get('url', '')}"[:280],
                f"Details: {summary[:100]}... Read more: {item.get('url', '')} #WorldAffairs"[:280]
            ]
        
        thread_json = {
            "thread": thread,
            "platform": "x",
            "char_limit": 280,
            "domain": domain
        }
        
        conn, cursor = get_db_connection()
        cursor.execute(
            "INSERT INTO threads (id, url, domain, content, thread_json, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), item["url"], domain, content[:1000], json.dumps(thread_json), time.strftime("%Y-%m-%dT%H:%M:%SZ"))
        )
        conn.commit()
        
        logger.info(f"Generated thread with {len(thread)} posts for {title[:50]} in domain {domain}")
        log_metric("threads_generated", 1)
        return thread, thread_json
    except Exception as e:
        logger.error(f"Error generating rich thread: {e}")
        return [f"Thread: {title[:50]} failed to generate. Source: {source}."[:280]], {"thread": [], "platform": "x", "domain": domain}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(tweepy.TweepyException))
def post_to_x(content, is_thread=False, parent_id=None):
    if config["test_mode"]:
        logger.info(f"Test mode: Simulated posting to X: {content[:50]}...")
        log_metric("posts_made", 1)
        return "mock_tweet_id"
    try:
        if is_thread and parent_id:
            tweet = client.create_tweet(text=content, in_reply_to_tweet_id=parent_id)
        else:
            tweet = client.create_tweet(text=content)
        logger.info(f"Posted to X: {content[:50]}... (ID: {tweet.data['id']})")
        log_metric("posts_made", 1)
        return tweet.data["id"]
    except tweepy.TweepyException as e:
        logger.error(f"Error posting to X: {e}")
        if e.response and e.response.status_code == 401:
            logger.error("401 Unauthorized. Verify X API credentials.")
        raise

@app.route('/generate_thread', methods=['POST'])
def generate_thread_api():
    try:
        data = request.get_json()
        url = data.get('url')
        domain = data.get('domain', 'geopolitics')
        if not url:
            return jsonify({"error": "URL is required"}), 400
        if domain not in config["domains"]:
            return jsonify({"error": f"Invalid domain. Choose from {list(config['domains'].keys())}"}), 400
        content = fetch_custom_url(url, domain)
        if not content:
            return jsonify({"error": "Failed to fetch content from URL"}), 500
        content["content_type"] = "custom"
        thread, thread_json = generate_rich_thread(content, domain)
        claim_id = str(uuid.uuid4())
        claim = content.get("transcript", content.get("content", content.get("title", "")))[:500]
        verified, sources, confidence = verify_claim(claim)
        conn, cursor = get_db_connection()
        cursor.execute(
            "INSERT INTO claims (id, source, claim, status, timestamp, content_type, post_content, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (claim_id, content["source"], claim, "pending", time.strftime("%Y-%m-%dT%H:%M:%SZ"), f"{domain}_thread", "\n".join(thread), confidence)
        )
        conn.commit()
        return jsonify({"thread": thread, "thread_json": thread_json, "claim_id": claim_id})
    except Exception as e:
        logger.error(f"Error in generate_thread_api: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/claims', methods=['GET'])
def get_pending_claims():
    try:
        conn, cursor = get_db_connection()
        cursor.execute("SELECT id, source, claim, content_type, post_content, confidence FROM claims WHERE status = 'pending'")
        claims = [
            {
                "id": row[0],
                "source": row[1],
                "claim": row[2],
                "content_type": row[3],
                "post_content": row[4].split("\n"),
                "confidence": row[5]
            } for row in cursor.fetchall()
        ]
        return jsonify({"claims": claims})
    except Exception as e:
        logger.error(f"Error fetching pending claims: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/approve_claim/<claim_id>', methods=['POST'])
def approve_claim(claim_id):
    try:
        conn, cursor = get_db_connection()
        cursor.execute("SELECT post_content, content_type FROM claims WHERE id = ? AND status = 'pending'", (claim_id,))
        result = cursor.fetchone()
        if not result:
            logger.error(f"Claim {claim_id} not found or not pending")
            return jsonify({"error": "Claim not found or not pending"}), 404
        post_content, content_type = result
        cursor.execute("UPDATE claims SET status = 'approved' WHERE id = ?", (claim_id,))
        conn.commit()
        posts = post_content.split("\n")
        parent_id = None
        for post in posts:
            if post.strip():
                parent_id = post_to_x(post, is_thread=bool(parent_id), parent_id=parent_id)
        logger.info(f"Approved and posted claim {claim_id}")
        log_metric("posts_approved", 1)
        return jsonify({"status": "approved", "claim_id": claim_id})
    except Exception as e:
        logger.error(f"Error approving claim {claim_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/reject_claim/<claim_id>', methods=['POST'])
def reject_claim(claim_id):
    try:
        conn, cursor = get_db_connection()
        cursor.execute("UPDATE claims SET status = 'rejected' WHERE id = ?", (claim_id,))
        conn.commit()
        logger.info(f"Rejected claim {claim_id}")
        log_metric("posts_rejected", 1)
        return jsonify({"status": "rejected", "claim_id": claim_id})
    except Exception as e:
        logger.error(f"Error rejecting claim {claim_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/logs', methods=['GET'])
def get_logs():
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = f.readlines()[-50:]
        return jsonify({"logs": logs})
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        return jsonify({"error": str(e)}), 500

def process_single_content(item, content_type):
    try:
        conn, cursor = get_db_connection()
        cursor.execute("SELECT content_id FROM processed_content WHERE content_id = ?", (item["id"],))
        if cursor.fetchone():
            logger.info(f"Content {item['id']} already processed")
            return
        claim = item.get("transcript", item.get("content", item.get("title", "")))[:500]
        claim_id = str(uuid.uuid4())
        verified, sources, confidence = verify_claim(claim)
        domain = item.get("domain", "geopolitics")
        item["content_type"] = content_type
        thread, _ = generate_rich_thread(item, domain)
        content_type_db = f"{domain}_thread" if content_type != "news" else "news_thread"
        post_content = "\n".join(thread)
        if verified and not is_sensitive(post_content) and content_type == "news":
            parent_id = None
            for post in thread:
                if post.strip():
                    parent_id = post_to_x(post, is_thread=bool(parent_id), parent_id=parent_id)
            cursor.execute(
                "INSERT INTO claims (id, source, claim, status, timestamp, content_type, post_content, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (claim_id, item["source"], claim, "approved", time.strftime("%Y-%m-%dT%H:%M:%SZ"), content_type_db, post_content, confidence)
            )
        else:
            cursor.execute(
                "INSERT INTO claims (id, source, claim, status, timestamp, content_type, post_content, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (claim_id, item["source"], claim, "pending", time.strftime("%Y-%m-%dT%H:%M:%SZ"), content_type_db, post_content, confidence)
            )
        cursor.execute(
            "INSERT INTO processed_content (id, source, content_id, timestamp) VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), item["source"], item["id"], item["timestamp"])
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Error processing content {item.get('id', 'unknown')}: {e}")

def process_content(exclude_youtube=False):
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for post in fetch_x_posts():
                futures.append(executor.submit(process_single_content, post, "x_post"))
            for news in fetch_news():
                futures.append(executor.submit(process_single_content, news, "news"))
            if not exclude_youtube:
                for video in fetch_youtube_videos():
                    futures.append(executor.submit(process_single_content, video, "youtube"))
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in parallel processing: {e}")
    except Exception as e:
        logger.error(f"Error in process_content: {e}")

async def schedule_tasks():
    schedule.every(config["schedule_interval_minutes"]).minutes.do(lambda: process_content(exclude_youtube=True))
    schedule.every(2).hours.do(process_content)  # YouTube every 2 hours
    while True:
        try:
            schedule.run_pending()
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Error in schedule_tasks: {e}")

async def run_flask():
    from wsgiref.simple_server import make_server
    server = make_server('127.0.0.1', 5001, app)
    logger.info("Starting Flask server on http://127.0.0.1:5001")
    await asyncio.get_event_loop().run_in_executor(None, server.serve_forever)

def signal_handler(sig, frame):
    logger.info(f"Received signal {sig}. Shutting down gracefully...")
    close_db_connection()
    loop.call_soon_threadsafe(loop.stop)
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()
    logger.info("Shutdown complete.")
    sys.exit(0)

async def main():
    logger.info("Starting news bot...")
    try:
        logger.info(f"Connected to X API as {user.username}")
        if config["test_mode"]:
            logger.info("Running in test mode")
        await asyncio.gather(
            run_flask(),
            schedule_tasks()
        )
    except Exception as e:
        logger.error(f"Main loop error: {e}")
        raise

if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    asyncio.run(main())