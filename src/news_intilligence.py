"""
FAST Incident Intelligence Collector + Geospatial Visualization
==============================================================
CrowdThreat-Level Geospatial Upgrade (Visualizer Integrated)

Main goals:
- Collect recent news via RSS/Atom feeds
- Filter & classify potential security/safety incidents
- Enrich with impact level and short factual summary
- Geocode mentioned locations when possible
- Display results + interactive map of incidents
"""

# ==============================
# IMPORTS
# ==============================

import json
import os
import random
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin

import dateparser
import pytz

import feedparser
import geonamescache
import pandas as pd
import requests
import streamlit as st
import trafilatura
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from tenacity import retry, stop_after_attempt, wait_exponential

# ─── Geospatial & Visualization dependencies ────────────────────────────────
import folium
import streamlit.components.v1 as components
from geotext import GeoText
from flashtext import KeywordProcessor
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter as GeoRateLimiter  # Renamed import to avoid name conflict

# ==============================
# CONFIGURATION
# ==============================

class Config:
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    USER_AGENT = "Mozilla/5.0 (IncidentCollectorFast/1.2)"
    DEFAULT_TIMEOUT = 12
    POLITE_SLEEP = 0.03
    GATE_TEXT_CHARS = 900
    DETAILS_TEXT_CHARS = 2500
    DEFAULT_EXTRACT_WORKERS = 8
    DEFAULT_LLM_WORKERS_OLLAMA = 2
    DEFAULT_LLM_WORKERS_GROQ = 4  # Increased from 1 for better parallelization
    LONDON_TIMEZONE = "Europe/London"
    ENABLE_LLM_CACHE = True  # Enable response caching for duplicate articles
    CACHE_MAX_SIZE = 1000  # Maximum number of cached responses
    COMMON_FEED_PATHS = ["/rss", "/rss.xml", "/feed", "/feed.xml", "/atom.xml", "/news/rss"]
    
    INCIDENT_KEYWORDS = re.compile(
        r"\b(shoot|shot|killed|dead|death|injur|wound|attack|explos|bomb|blast|fire|"
        r"arson|robber|stabb|rape|protest|riot|clash|crash|collision|"
        r"ship|vessel|maritime|piracy|hijack|aircraft|plane|helicopter|"
        r"police|arrest|suspect|gun|knife)\b",
        re.IGNORECASE,
    )


# ==============================
# UTILITY CLASSES
# ==============================

class TextUtils:
    @staticmethod
    def normalize_url(u: str) -> str:
        try:
            p = urlparse(u)
            return p._replace(fragment="").geturl()
        except Exception:
            return u

    @staticmethod
    def remove_think_blocks(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def normalize_nulls(text: str) -> str:
        t = text
        t = re.sub(r"\bNULL\b", "null", t, flags=re.IGNORECASE)
        t = re.sub(r"\bNone\b", "null", t, flags=re.IGNORECASE)
        return t

    @staticmethod
    def safe_parse_json(text: str) -> Tuple[Optional[dict], Optional[str]]:
        try:
            cleaned = TextUtils.normalize_nulls(TextUtils.remove_think_blocks(text))
            return json.loads(cleaned), None
        except Exception as e:
            return None, str(e)


class LLMCache:
    """LRU cache for LLM responses based on article content hash."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_order: deque = deque()
        self.hits = 0
        self.misses = 0
    
    def _hash_content(self, text: str) -> str:
        """Generate hash from article content for cache key."""
        import hashlib
        # Use first 1000 chars to generate hash (enough to identify unique articles)
        content_sample = text[:1000].strip().lower()
        return hashlib.sha256(content_sample.encode('utf-8')).hexdigest()
    
    def get(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve cached analysis for article."""
        text = f"{article.get('title', '')} {article.get('text', '')}"
        key = self._hash_content(text)
        
        if key in self.cache:
            self.hits += 1
            # Update access order (move to end for LRU)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, article: Dict[str, Any], result: Dict[str, Any]):
        """Store analysis result in cache."""
        text = f"{article.get('title', '')} {article.get('text', '')}"
        key = self._hash_content(text)
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest = self.access_order.popleft()
            del self.cache[oldest]
        
        self.cache[key] = result
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate_percent": round(hit_rate, 1),
            "cache_size": len(self.cache),
        }


class TimeUtils:
    """Time parsing and timezone conversion utilities."""
    
    @staticmethod
    def extract_time_phrases(text: str) -> List[str]:
        """Extract potential time-related phrases from text."""
        phrases = []
        
        # Pattern for relative times: "X hours/days/minutes ago"
        relative_pattern = r'\b\d+\s+(?:second|minute|hour|day|week|month|year)s?\s+ago\b'
        phrases.extend(re.findall(relative_pattern, text, re.IGNORECASE))
        
        # Pattern for "yesterday", "today", "this morning", etc.
        temporal_words = r'\b(?:yesterday|today|tonight|this\s+(?:morning|afternoon|evening)|last\s+(?:night|week|month))\b'
        phrases.extend(re.findall(temporal_words, text, re.IGNORECASE))
        
        # Pattern for specific times: "at 14:30", "9:00 AM", etc.
        time_pattern = r'\b(?:at\s+)?\d{1,2}:\d{2}(?:\s*(?:AM|PM|GMT|UTC))?\b'
        phrases.extend(re.findall(time_pattern, text, re.IGNORECASE))
        
        # Pattern for dates: "February 6", "6 Feb 2026", etc.
        date_pattern = r'\b(?:\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b'
        phrases.extend(re.findall(date_pattern, text, re.IGNORECASE))
        
        return phrases
    
    @staticmethod
    def parse_and_format_time(text: str, current_time: datetime) -> Optional[str]:
        """
        Parse time information from text and return in HH:MM:SS format (London time).
        
        Args:
            text: Text containing time information (may include relative times like "4 hours ago")
            current_time: Current UTC time for calculating relative times
            
        Returns:
            Time string in HH:MM:SS format (London timezone) or None if no time found
        """
        if not text or not isinstance(text, str):
            return None
        
        try:
            # First, extract time-related phrases
            time_phrases = TimeUtils.extract_time_phrases(text)
            
            # Try parsing each phrase
            for phrase in time_phrases:
                try:
                    parsed_time = dateparser.parse(
                        phrase,
                        settings={
                            'RELATIVE_BASE': current_time,
                            'RETURN_AS_TIMEZONE_AWARE': True,
                            'PREFER_DATES_FROM': 'past',
                        }
                    )
                    
                    if parsed_time:
                        # Convert to London timezone
                        london_tz = pytz.timezone(Config.LONDON_TIMEZONE)
                        
                        # If the parsed time is naive, assume UTC
                        if parsed_time.tzinfo is None:
                            parsed_time = pytz.utc.localize(parsed_time)
                        
                        # Convert to London time
                        london_time = parsed_time.astimezone(london_tz)
                        
                        # Return in HH:MM:SS format
                        return london_time.strftime("%H:%M:%S")
                except Exception:
                    continue
            
            # If no phrases found, try parsing the first 200 chars directly
            short_text = text[:200]
            parsed_time = dateparser.parse(
                short_text,
                settings={
                    'RELATIVE_BASE': current_time,
                    'RETURN_AS_TIMEZONE_AWARE': True,
                    'PREFER_DATES_FROM': 'past',
                }
            )
            
            if parsed_time:
                london_tz = pytz.timezone(Config.LONDON_TIMEZONE)
                if parsed_time.tzinfo is None:
                    parsed_time = pytz.utc.localize(parsed_time)
                london_time = parsed_time.astimezone(london_tz)
                return london_time.strftime("%H:%M:%S")
            
            return None
            
        except Exception:
            return None



class GeoUtils:
    _gc = geonamescache.GeonamesCache()
    _cities = _gc.get_cities()
    _countries = _gc.get_countries()

    @classmethod
    def city_to_country(cls, city_name: str) -> Optional[str]:
        if not city_name:
            return None
        target = city_name.strip().lower()
        for _, c in cls._cities.items():
            if (c.get("name") or "").strip().lower() == target:
                cc = c.get("countrycode")
                if cc and cc in cls._countries:
                    return cls._countries[cc]["name"]
        return None


class TpmRateLimiter:  # Renamed to avoid conflict with geopy's RateLimiter
    def __init__(self, tpm_limit: int):
        self.tpm_limit = tpm_limit
        self.window = deque()

    def _cleanup(self):
        now = time.time()
        while self.window and (now - self.window[0][0] > 60):
            self.window.popleft()

    def used(self) -> int:
        self._cleanup()
        return sum(t for _, t in self.window)

    def wait_for_budget(self, tokens_needed: int):
        while True:
            self._cleanup()
            if self.used() + tokens_needed <= self.tpm_limit:
                self.window.append((time.time(), tokens_needed))
                return
            time.sleep(0.2)


class NetworkUtils:
    @staticmethod
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.6, min=0.4, max=2))
    def http_get(url: str, timeout: int = Config.DEFAULT_TIMEOUT) -> requests.Response:
        headers = {"User-Agent": Config.USER_AGENT}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r


# ==============================
# GEOSPATIAL VISUALIZATION CLASS  (NEW)
# ==============================

class IncidentVisualizer:
    """Handles location extraction and Folium map rendering in Streamlit."""

    def __init__(self, user_agent="incident_intel_v6"):
        # Initialize Geospatial Parser
        self.geolocator = Nominatim(user_agent=user_agent)
        self.geocode_service = GeoRateLimiter(self.geolocator.geocode, min_delay_seconds=1.5, swallow_exceptions=True)
        self.keyword_processor = KeywordProcessor(case_sensitive=False)
        
        # Add infrastructure keywords for better landmark matching
        infra = ["airport", "station", "bridge", "highway", "mall", "hospital", "port", "square", "embassy"]
        for k in infra:
            self.keyword_processor.add_keyword(k)

    def get_geospatial_data(self, text: str, country_hint: Optional[str] = None, city_hint: Optional[str] = None) -> Dict[str, Any]:
        """Extracts city, address / landmark and coordinates from text."""
        places = GeoText(text)
        city = city_hint or (places.cities[0] if places.cities else None)
        
        # Regex for common street / avenue patterns
        street_match = re.search(r'\b[A-Z][a-z]+\s(?:Street|St|Road|Rd|Ave|Blvd)\b', text)
        landmark = self.keyword_processor.extract_keywords(text)
        
        site = (landmark[0] if landmark else None) or (street_match.group(0) if street_match else None)
        
        # Build geocode query
        query_parts = [p for p in [site, city, country_hint] if p]
        query = ", ".join(query_parts) if query_parts else None
        
        lat, lon = None, None
        if query:
            loc = self.geocode_service(query)
            if loc:
                lat, lon = loc.latitude, loc.longitude
        
        # Fallback: if address is None but city and country exist, geocode city center
        if site is None and lat is None and lon is None and city and country_hint:
            try:
                city_query = f"{city}, {country_hint}"
                city_loc = self.geocode_service(city_query)
                if city_loc:
                    lat, lon = city_loc.latitude, city_loc.longitude
                    # Use city name as approximate address
                    site = city
            except Exception:
                pass  # Gracefully handle geocoding errors
        
        return {"city": city, "address": site, "lat": lat, "lon": lon}

    def create_and_render_map(self, df: pd.DataFrame, height: int = 500) -> bool:
        """Generates Folium map and renders it in Streamlit."""
        map_data = df.dropna(subset=['latitude', 'longitude']).copy()
        
        if map_data.empty:
            return False

        # Center map on mean coordinates of all valid points
        m = folium.Map(
            location=[map_data['latitude'].mean(), map_data['longitude'].mean()], 
            zoom_start=6
        )
        
        colors = {"High": "red", "Medium": "orange", "Low": "green"}
        
        for _, row in map_data.iterrows():
            impact_color = colors.get(row.get('impact'), "gray")
            
            # Sanitize strings for safe HTML popup
            from html import escape
            title = escape(str(row.get('title', 'N/A')))
            summary = escape(str(row.get('summary', 'No summary available'))[:400])
            url = str(row.get('url', ''))
            incident_type = escape(str(row.get('incident_type', 'N/A')))
            
            # Create enhanced popup with title, summary, and clickable URL (all 12pt font)
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; font-size: 12pt;">
                <b>{title}</b><br><br>
                <b>Type:</b> {incident_type}<br>
                <b>Impact:</b> {row.get('impact', 'N/A')}<br><br>
                <b>Summary:</b><br>
                {summary}<br><br>
                <a href="{url}" target="_blank" style="font-size: 12pt;">View Article</a>
            </div>
            """
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=7,
                color=impact_color,
                fill=True,
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=400),
                tooltip=str(row.get('title', 'Incident'))
            ).add_to(m)
        
        # Render map as static HTML component
        map_html = m._repr_html_()
        components.html(map_html, height=height)
        return True


# ==============================
# CORE BUSINESS LOGIC (unchanged except final enrichment)
# ==============================

class FeedDiscoverer:
    @staticmethod
    @st.cache_data(ttl=60*60)
    def discover_best_feed(home_url: str) -> Optional[str]:
        base = home_url.rstrip("/")
        try:
            r = NetworkUtils.http_get(base)
            soup = BeautifulSoup(r.text, "html.parser")
            feed_links = []
            for link in soup.find_all("link", attrs={"rel": True, "href": True}):
                rel = " ".join(link.get("rel") or []).lower()
                typ = (link.get("type") or "").lower()
                href = link.get("href")
                if "alternate" in rel and ("rss" in typ or "atom" in typ):
                    feed_links.append(urljoin(base + "/", href))
            for f in feed_links:
                if FeedDiscoverer.parse_feed(f, max_items=3):
                    return f
        except Exception:
            pass

        for p in Config.COMMON_FEED_PATHS:
            f = base + p
            if FeedDiscoverer.parse_feed(f, max_items=3):
                return f
        return None

    @staticmethod
    @st.cache_data(ttl=5*60)
    def parse_feed(feed_url: str, max_items: int = 10) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        try:
            parsed = feedparser.parse(feed_url)
            for e in (parsed.entries or [])[:max_items]:
                link = getattr(e, "link", None)
                if not link:
                    continue
                items.append({
                    "title": getattr(e, "title", "") or "",
                    "url": TextUtils.normalize_url(link),
                    "published": getattr(e, "published", None) or getattr(e, "updated", None),
                })
        except Exception:
            pass
        return items

    @staticmethod
    def collect_links(selected_sources: List[Dict[str, str]], max_items_per_source: int) -> List[Dict[str, Any]]:
        all_links: List[Dict[str, Any]] = []
        for s in selected_sources:
            source_name = s["source"]
            home = s["link"].rstrip("/")
            feed = FeedDiscoverer.discover_best_feed(home)
            if not feed:
                continue
            items = FeedDiscoverer.parse_feed(feed, max_items=max_items_per_source)
            for it in items:
                all_links.append({
                    "source": source_name,
                    "url": it["url"],
                    "feed_title": it.get("title", ""),
                    "published": it.get("published"),
                })
            time.sleep(Config.POLITE_SLEEP)

        # Deduplicate
        seen = set()
        deduped = []
        for row in all_links:
            if row["url"] not in seen:
                seen.add(row["url"])
                deduped.append(row)
        return deduped


class ArticleExtractor:
    @staticmethod
    def extract_one(url: str) -> Dict[str, Any]:
        out = {
            "ok": False, 
            "url": url, 
            "title": "", 
            "text": "",
            "id": None,  # fingerprint
            # "description": "",  # excerpt
            "source": "",  # source hostname
            "author": "",
            "date_published": None,
            "date_modified": None,
            "categories": None,
            "tags": None,
            "error": None,
        }
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                out["error"] = "fetch_url empty"
                return out
            
            # Extract with full metadata including comments set to False
            extracted = trafilatura.extract(
                downloaded, 
                output_format="json", 
                include_comments=False,
                with_metadata=True
            )
            if not extracted:
                out["error"] = "extract empty"
                return out
            
            data = json.loads(extracted)
            
            # Map Trafilatura fields to output fields
            out["title"] = (data.get("title") or "").strip()
            out["text"] = (data.get("text") or "").strip()
            out["id"] = data.get("fingerprint")  # Unique content fingerprint
            # out["description"] = (data.get("excerpt") or "").strip()
            #out["source"] = (data.get("source-hostname") or "").strip()  # Source hostname from Trafilatura
            out["author"] = (data.get("author") or "").strip()
            out["date_published"] = data.get("date") or data.get("date_published")
            out["date_modified"] = data.get("date_modified")
            out["categories"] = data.get("categories")  # Can be list or None
            out["tags"] = data.get("tags")  # Can be list or None
            
            out["ok"] = bool(out["text"])
            return out
        except Exception as e:
            out["error"] = str(e)
            return out

    @staticmethod
    def extract_parallel(links: List[Dict[str, Any]], workers: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_map = {ex.submit(ArticleExtractor.extract_one, row["url"]): row for row in links}
            for fut in as_completed(future_map):
                meta = future_map[fut]
                try:
                    art = fut.result()
                    art["source"] = meta["source"]
                    art["feed_published"] = meta.get("published")
                    art["feed_title"] = meta.get("feed_title", "")
                    out.append(art)
                except Exception:
                    continue
        return [a for a in out if a.get("ok")]


class IncidentAnalyzer:
    def __init__(
        self,
        backend: str,
        ollama_model: str,
        groq_model: str,
        temperature: float,
        groq_tpm_limit: int = 6000,
        enable_cache: bool = True,
    ):
        self.backend = backend
        self.llm = self._build_llm(backend, ollama_model, groq_model, temperature)
        self.text_parser = StrOutputParser()
        self.groq_bucket = TpmRateLimiter(groq_tpm_limit) if backend == "Groq (cloud)" else None
        self.cache = LLMCache(max_size=Config.CACHE_MAX_SIZE) if enable_cache else None
        self._init_prompts_and_chains()

    def _build_llm(self, backend, ollama_model, groq_model, temperature):
        if backend == "Ollama (local, free)":
            return ChatOllama(model=ollama_model, temperature=temperature)
        return ChatGroq(model=groq_model, temperature=temperature)

    def _init_prompts_and_chains(self):
        # Unified prompt that returns all fields in one JSON response
        self.unified_prompt = ChatPromptTemplate.from_template(
            """
You are an incident analyst. Analyze the following article and return ONLY valid JSON (no markdown, no extra text).

Schema:
{{
  "incident_type": string,
  "impact": string,
  "location": {{
    "country": string|null,
    "city": string|null,
    "address": string|null
  }},
  "summary": string
}}

Instructions:
1. incident_type: Classify into exactly ONE category: criminality, drugs, stabbing, robbery, bombing, arson, protest, maritime, rape, air, traffic, or Other
2. impact: One of: High (fatalities OR national security risk OR mass casualties), Medium (injuries OR localized disruption, no fatalities), or Low (minor, no casualties, minimal disruption)
3. location: Extract country, city, and address/landmark if mentioned. Use null for unknown fields.
4. summary: Exactly 5-6 sentences, factual, no speculation

Text:
{context}
"""
        )
        self.unified_chain = self.unified_prompt | self.llm | self.text_parser

    def _estimate_tokens(self, text: str) -> int:
        return max(250, int(len(text) / 4))

    def _invoke_with_limits(self, chain, payload: Dict[str, Any], max_retries: int = 6):
        if self.backend != "Groq (cloud)":
            return chain.invoke(payload)

        ctx = payload.get("context", "") or ""
        self.groq_bucket.wait_for_budget(self._estimate_tokens(ctx))

        for attempt in range(max_retries):
            try:
                return chain.invoke(payload)
            except Exception as e:
                msg = str(e).lower()
                if "rate_limit" in msg or "429" in msg or "tpm" in msg:
                    sleep_s = min(4.0, (0.5 * (2**attempt))) + random.uniform(0, 0.2)
                    time.sleep(sleep_s)
                    continue
                raise
        raise RuntimeError("Groq rate-limit retries exceeded")

    def analyze_unified(self, article: Dict[str, Any], keyword_prefilter: bool) -> Dict[str, Any]:
        """Unified analysis that gets all fields in a single LLM call."""
        # Check cache first
        if self.cache:
            cached = self.cache.get(article)
            if cached:
                # Apply cached results to article
                article.update(cached)
                return article
        
        # Keyword pre-filter for fast rejection
        title = article.get("title") or article.get("feed_title") or ""
        snippet = (article.get("text") or "")[:400]
        
        if keyword_prefilter:
            combined_text = f"{title} {snippet}"
            if not Config.INCIDENT_KEYWORDS.search(combined_text):
                result = {
                    "incident_type": "Other",
                    "impact": None,
                    "country": None,
                    "city": None,
                    "address": None,
                    "location_json": json.dumps({"country": None, "city": None, "address": None}),
                    "summary": "",
                }
                article.update(result)
                if self.cache:
                    self.cache.put(article, result)
                return article
        
        # Build context for unified analysis
        text = article.get("text") or ""
        ctx = (title + "\n\n" + text[:Config.DETAILS_TEXT_CHARS]).strip()
        
        # Single LLM call for all fields
        raw = self._invoke_with_limits(self.unified_chain, {"context": ctx})
        obj, err = TextUtils.safe_parse_json(raw)
        
        if obj is None:
            # Fallback on parse error
            obj = {
                "incident_type": "Other",
                "impact": None,
                "location": {"country": None, "city": None, "address": None, "error": err},
                "summary": "",
            }
        
        # Extract and normalize fields
        incident_type = (obj.get("incident_type") or "Other").strip()
        
        impact = (obj.get("impact") or "").strip().capitalize()
        if impact not in {"High", "Medium", "Low"}:
            impact = None
        
        loc = obj.get("location") or {"country": None, "city": None, "address": None}
        
        # Clean empty strings
        for k in ["country", "city", "address"]:
            if k in loc and isinstance(loc[k], str) and not loc[k].strip():
                loc[k] = None
        
        # Infer country from city if possible
        if loc.get("city") and not loc.get("country"):
            inferred = GeoUtils.city_to_country(loc["city"])
            if inferred:
                loc["country"] = inferred
        
        # Build result
        result = {
            "incident_type": incident_type,
            "impact": impact,
            "country": loc.get("country"),
            "city": loc.get("city"),
            "address": loc.get("address"),
            "location_json": json.dumps(loc, ensure_ascii=False),
            "summary": obj.get("summary") or "",
        }
        
        # Cache the result
        if self.cache:
            self.cache.put(article, result)
        
        # Update article with results
        article.update(result)
        return article

    def process_parallel(
        self,
        articles: List[Dict[str, Any]],
        stage: str,
        workers: int,
        keyword_prefilter: bool = False,
    ) -> List[Dict[str, Any]]:
        if not articles:
            return articles

        process_list = articles
        
        # For unified stage, process all articles
        # For legacy stages (if any), filter out "Other" incidents
        if stage in ["impact", "details"]:
            process_list = [
                a for a in articles
                if (a.get("incident_type") or "").strip().lower() != "other"
            ]

        if not process_list:
            return articles

        with ThreadPoolExecutor(max_workers=workers) as ex:
            if stage == "unified":
                futs = [ex.submit(self.analyze_unified, a, keyword_prefilter) for a in process_list]
            elif stage == "gate":
                # Legacy support (shouldn't be used anymore)
                futs = [ex.submit(self.analyze_unified, a, keyword_prefilter) for a in process_list]
            elif stage == "impact":
                # Legacy support (shouldn't be used anymore)
                futs = [ex.submit(self.analyze_unified, a, False) for a in process_list]
            elif stage == "details":
                # Legacy support (shouldn't be used anymore)
                futs = [ex.submit(self.analyze_unified, a, False) for a in process_list]
            else:
                return articles

            for _ in as_completed(futs):
                pass
        
        return articles
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics if caching is enabled."""
        return self.cache.get_stats() if self.cache else None


# ==============================
# MAIN APPLICATION
# ==============================

def main():
    st.set_page_config(page_title="⚡ Incident Collector + Map", layout="wide")
    st.title("⚡ Incident Intelligence Collector  —  Geospatial View")

    st.markdown(
        """
        Collects recent news → filters security/safety incidents → classifies & summarizes  
        → attempts to geocode locations → shows table + **interactive map**
        """
    )

    # ─── 1. Source selection ────────────────────────────────────────────────
    st.subheader("1. Source Configuration")

    default_csv = r"C:\\Yasser-SSD\\Langchain_Projects_01012026\\RFP_Proposal_Generator\\list_of_news_sources.csv"
    csv_path = st.text_input("Path to Sources CSV", value=default_csv)

    if not os.path.exists(csv_path):
        st.error(f"CSV not found: {csv_path}")
        st.stop()

    sources_df = pd.read_csv(csv_path)
    sources_df.columns = [c.lower() for c in sources_df.columns]

    if not {"source", "link"}.issubset(set(sources_df.columns)):
        st.error("CSV must contain columns: source, link")
        st.stop()

    q = st.text_input("Filter sources", value="").strip().lower()
    filtered = sources_df
    if q:
        filtered = filtered[
            filtered["source"].astype(str).str.lower().str.contains(q)
            | filtered["link"].astype(str).str.lower().str.contains(q)
        ]

    labels = [f"{row['source']} — {row['link']}" for _, row in filtered.iterrows()]
    selected_labels = st.multiselect(
        "Select sources", options=labels, default=labels[:5] if labels else []
    )

    selected_sources = []
    for lbl in selected_labels:
        name, link = lbl.split(" — ", 1)
        selected_sources.append({"source": name.strip(), "link": link.strip()})

    # ─── 2. Pipeline settings ───────────────────────────────────────────────
    st.subheader("2. Pipeline Settings")
    colA, colB, colC, colD = st.columns(4)
    with colA:
        max_items = st.number_input("Items / Source", 1, 500, 10)
    with colB:
        max_total = st.number_input("Total Cap", 1, 500, 50)
    with colC:
        do_keyword = st.checkbox("Keyword Pre-filter", value=True)
    with colD:
        extract_workers = st.number_input(
            "Extract Workers", 1, 20, Config.DEFAULT_EXTRACT_WORKERS
        )

    # ─── 3. LLM settings ────────────────────────────────────────────────────
    st.subheader("3. Model Settings")
    backend = st.selectbox(
        "Backend", ["Ollama (local, free)", "Groq (cloud)"], index=1
    )

    c1, c2, c3 = st.columns(3)
    with c1: temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    with c2: ollama_mod = st.text_input("Ollama Model", "qwen2.5:7b-instruct")
    with c3: groq_mod  = st.text_input("Groq Model", "qwen/qwen3-32b")

    groq_bucket_limit = 6000
    llm_workers_def = (
        Config.DEFAULT_LLM_WORKERS_OLLAMA if backend.startswith("Ollama")
        else Config.DEFAULT_LLM_WORKERS_GROQ
    )
    
    c4, c5 = st.columns(2)
    with c4:
        llm_workers = st.number_input("LLM Workers", 1, 10, llm_workers_def)
    with c5:
        if backend.startswith("Groq"):
            groq_bucket_limit = st.number_input("Groq TPM Limit", 1000, 20000, 6000, 500)

    # ─── 4. Run button & pipeline ───────────────────────────────────────────
    if st.button("🚀 Run Pipeline"):
        if not selected_sources:
            st.warning("Please select at least one source.")
            st.stop()

        start_time = time.time()

        # Initialize LLM analyzer
        analyzer = IncidentAnalyzer(
            backend=backend,
            ollama_model=ollama_mod.strip(),
            groq_model=groq_mod.strip(),
            temperature=float(temp),
            groq_tpm_limit=int(groq_bucket_limit),
            enable_cache=Config.ENABLE_LLM_CACHE,
        )

        # ── Step 1: Collect RSS/Atom links ────────────────────────────────
        with st.spinner("Collecting links..."):
            links = FeedDiscoverer.collect_links(selected_sources, int(max_items))
            links = links[:int(max_total)]
        st.write(f"Collected URLs: **{len(links)}**")

        # ── Step 2: Extract article content ───────────────────────────────
        with st.spinner(f"Extracting articles ({extract_workers} workers)..."):
            articles = ArticleExtractor.extract_parallel(links, int(extract_workers))
        st.write(f"Extracted articles: **{len(articles)}**")

        # ── Step 3: Unified Analysis (Single LLM call per article) ─────────
        with st.spinner(f"Analyzing incidents (unified pipeline, {llm_workers} workers)..."):
            articles = analyzer.process_parallel(
                articles, "unified", int(llm_workers), keyword_prefilter=do_keyword
            )
        
        # ── Final filtering ───────────────────────────────────────────────
        results = [
            a for a in articles
            if (a.get("incident_type") or "").strip().lower() != "other"
        ]
        st.write(f"Incident candidates: **{len(results)}**")

        elapsed_time = time.time() - start_time
        st.success(f"✅ Finished in {elapsed_time:.1f} s")
        
        # Display cache statistics if caching is enabled
        if Config.ENABLE_LLM_CACHE:
            cache_stats = analyzer.get_cache_stats()
            if cache_stats:
                st.info(
                    f"📊 Cache Stats: {cache_stats['hits']} hits, {cache_stats['misses']} misses, "
                    f"{cache_stats['hit_rate_percent']}% hit rate, {cache_stats['cache_size']} cached entries"
                )
        
        if not results:
            st.warning("No incidents found.")
            st.stop()

        # ── Prepare clean DataFrame ───────────────────────────────────────
        # Get current time for relative time calculations
        current_utc_time = datetime.utcnow()
        
        results_clean = []
        for a in results:
            # Extract time information from article content
            time_text = " ".join([
                str(a.get("title") or ""),
                str(a.get("text", ""))[:500],  # First 500 chars for time extraction
                str(a.get("summary") or ""),
            ])
            incident_time = TimeUtils.parse_and_format_time(time_text, current_utc_time)
            
            item = {
                "id": a.get("id"),  # Fingerprint
                "source": a.get("source"),
                #"source": a.get("source-hostname"),  # Hostname from Trafilatura
                "url": a.get("url"),
                "title": a.get("title") or a.get("feed_title"),
                #"description": a.get("excerpt"),  # Description from Trafilatura
                "text": a.get("text", ""),  # Full article text
                "author": a.get("author"),
                "date": a.get("date_published") or a.get("feed_published"),
                "date_modified": a.get("date_modified"),
                "time": incident_time,  # New time column in HH:MM:SS format
                "categories": a.get("categories"),
                "tags": a.get("tags"),
                "type": a.get("incident_type"),
                "impact": a.get("impact"),
                "country": a.get("country"),
                "city": a.get("city"),
                "address": a.get("address"),
                "summary": a.get("summary"),
            }
            results_clean.append(item)

        df = pd.DataFrame(results_clean)

        # ── NEW: Geospatial enrichment ─────────────────────────────────────
        st.subheader("Geolocation Enrichment")
        with st.spinner("Trying to geocode incident locations..."):
            viz = IncidentVisualizer()
            geo_results = []

            for _, row in df.iterrows():
                text_for_geo = " ".join([
                    str(row.get("title", "")),
                    str(row.get("summary", "")),
                    str(row.get("address", "") or ""),
                ])
                geo = viz.get_geospatial_data(
                    text_for_geo,
                    country_hint=row.get("country"),
                    city_hint=row.get("city")  # Pass city hint for better matching
                )
                geo_results.append(geo)

            # Merge coordinates back to DataFrame
            df["latitude"]  = [g["lat"] for g in geo_results]
            df["longitude"] = [g["lon"] for g in geo_results]

        # ── Display results table ──────────────────────────────────────────
        st.subheader("Detected Incidents")
        st.dataframe(df, use_container_width=True)

        # ── NEW: Interactive map ───────────────────────────────────────────
        st.subheader("🗺️ Incident Map")
        success = viz.create_and_render_map(df, height=600)

        if not success:
            st.info("No incidents could be placed on the map (missing coordinates).")

        # ── Export ─────────────────────────────────────────────────────────
        c_csv, c_json = st.columns(2)
        with c_csv:
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                "incidents.csv",
                "text/csv",
            )
        with c_json:
            st.download_button(
                "Download JSON",
                json.dumps(results_clean, indent=2, ensure_ascii=False).encode("utf-8"),
                "incidents.json",
                "application/json",
            )


if __name__ == "__main__":
    main()



# """
# FAST Incident Intelligence Collector
# ====================================

# Design Goals
# ------------
# - Finish ~10 articles in < 1 minute in typical conditions.
# - Default to FREE local inference (Ollama) to avoid Groq rate limits.
# - Allow optional Groq backend with TPM throttling + 429 retry.
# - Skip processing details if incident type is "Other".

# Key Performance Techniques
# --------------------------
# 1) Cached RSS/Atom feed discovery per domain (Streamlit cache).
# 2) Parallel article extraction (ThreadPoolExecutor).
# 3) Modular LLM pipeline (Gate -> Impact -> Details).
# 4) Optional keyword pre-filter.

# """

# import json
# import os
# import random
# import re
# import time
# from collections import deque
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from typing import Any, Dict, List, Optional, Tuple, Union
# from urllib.parse import urljoin, urlparse

# import feedparser
# import geonamescache
# import pandas as pd
# import requests
# import streamlit as st
# import trafilatura
# from bs4 import BeautifulSoup
# from dotenv import load_dotenv
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_groq import ChatGroq
# from langchain_ollama import ChatOllama
# from tenacity import retry, stop_after_attempt, wait_exponential

# # ==============================
# # CONFIGURATION & CONSTANTS
# # ==============================


# class Config:
#     """Application configuration and constants."""

#     load_dotenv()

#     # Environment
#     GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

#     # Network
#     USER_AGENT = "Mozilla/5.0 (IncidentCollectorFast/1.2)"
#     DEFAULT_TIMEOUT = 12
#     POLITE_SLEEP = 0.03

#     # Text Processing Limits
#     GATE_TEXT_CHARS = 900
#     DETAILS_TEXT_CHARS = 2500

#     # Default Concurrency
#     DEFAULT_EXTRACT_WORKERS = 8
#     DEFAULT_LLM_WORKERS_OLLAMA = 2
#     DEFAULT_LLM_WORKERS_GROQ = 1

#     # Feed Discovery
#     COMMON_FEED_PATHS = [
#         "/rss",
#         "/rss.xml",
#         "/feed",
#         "/feed.xml",
#         "/atom.xml",
#         "/news/rss",
#     ]

#     # Incident Pre-filter Keywords
#     INCIDENT_KEYWORDS = re.compile(
#         r"\b(shoot|shot|killed|dead|death|injur|wound|attack|explos|bomb|blast|fire|"
#         r"arson|robber|stabb|rape|protest|riot|clash|crash|collision|"
#         r"ship|vessel|maritime|piracy|hijack|aircraft|plane|helicopter|"
#         r"police|arrest|suspect|gun|knife)\b",
#         re.IGNORECASE,
#     )


# # ==============================
# # UTILITY CLASSES
# # ==============================


# class TextUtils:
#     """Text processing and normalization utilities."""

#     @staticmethod
#     def normalize_url(u: str) -> str:
#         """Remove URL fragments (#...) while keeping query parameters."""
#         try:
#             p = urlparse(u)
#             return p._replace(fragment="").geturl()
#         except Exception:
#             return u

#     @staticmethod
#     def remove_think_blocks(text: str) -> str:
#         """Remove <think>...</think> blocks if the model outputs them."""
#         return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

#     @staticmethod
#     def normalize_nulls(text: str) -> str:
#         """Convert NULL/None to JSON null to reduce json.loads errors."""
#         t = text
#         t = re.sub(r"\\bNULL\\b", "null", t, flags=re.IGNORECASE)
#         t = re.sub(r"\\bNone\\b", "null", t, flags=re.IGNORECASE)
#         return t

#     @staticmethod
#     def safe_parse_json(text: str) -> Tuple[Optional[dict], Optional[str]]:
#         """Safely parse JSON from model output."""
#         try:
#             cleaned = TextUtils.normalize_nulls(TextUtils.remove_think_blocks(text))
#             return json.loads(cleaned), None
#         except Exception as e:
#             return None, str(e)


# class GeoUtils:
#     """Geographic data utilities."""

#     _gc = geonamescache.GeonamesCache()
#     _cities = _gc.get_cities()
#     _countries = _gc.get_countries()

#     @classmethod
#     def city_to_country(cls, city_name: str) -> Optional[str]:
#         """
#         Best-effort offline mapping from city name to country using geonamescache.
#         """
#         if not city_name:
#             return None
#         target = city_name.strip().lower()
#         for _, c in cls._cities.items():
#             if (c.get("name") or "").strip().lower() == target:
#                 cc = c.get("countrycode")
#                 if cc and cc in cls._countries:
#                     return cls._countries[cc]["name"]
#         return None


# class RateLimiter:
#     """Rolling-window token-per-minute limiter (simple)."""

#     def __init__(self, tpm_limit: int):
#         self.tpm_limit = tpm_limit
#         self.window = deque()  # list of (timestamp, tokens)

#     def _cleanup(self):
#         now = time.time()
#         while self.window and (now - self.window[0][0] > 60):
#             self.window.popleft()

#     def used(self) -> int:
#         self._cleanup()
#         return sum(t for _, t in self.window)

#     def wait_for_budget(self, tokens_needed: int):
#         """ detailed docstring"""
#         while True:
#             self._cleanup()
#             if self.used() + tokens_needed <= self.tpm_limit:
#                 self.window.append((time.time(), tokens_needed))
#                 return
#             time.sleep(0.2)


# class NetworkUtils:
#     """HTTP networking utilities."""

#     @staticmethod
#     @retry(
#         stop=stop_after_attempt(2),
#         wait=wait_exponential(multiplier=0.6, min=0.4, max=2),
#     )
#     def http_get(url: str, timeout: int = Config.DEFAULT_TIMEOUT) -> requests.Response:
#         """HTTP GET with short retries."""
#         headers = {"User-Agent": Config.USER_AGENT}
#         r = requests.get(url, headers=headers, timeout=timeout)
#         r.raise_for_status()
#         return r


# # ==============================
# # CORE BUSINESS LOGIC
# # ==============================


# class FeedDiscoverer:
#     """Handles RSS/Atom feed discovery and parsing."""

#     @staticmethod
#     @st.cache_data(ttl=60 * 60)
#     def discover_best_feed(home_url: str) -> Optional[str]:
#         """
#         Discover the best RSS/Atom feed URL for a given site.
#         Cached 1 hour.
#         """
#         base = home_url.rstrip("/")

#         # 1) Homepage discovery
#         try:
#             r = NetworkUtils.http_get(base)
#             soup = BeautifulSoup(r.text, "html.parser")

#             feed_links = []
#             for link in soup.find_all("link", attrs={"rel": True, "href": True}):
#                 rel = " ".join(link.get("rel") or []).lower()
#                 typ = (link.get("type") or "").lower()
#                 href = link.get("href")
#                 if "alternate" in rel and ("rss" in typ or "atom" in typ):
#                     feed_links.append(urljoin(base + "/", href))

#             for f in feed_links:
#                 if FeedDiscoverer.parse_feed(f, max_items=3):
#                     return f
#         except Exception:
#             pass

#         # 2) Common endpoints
#         for p in Config.COMMON_FEED_PATHS:
#             f = base + p
#             if FeedDiscoverer.parse_feed(f, max_items=3):
#                 return f

#         return None

#     @staticmethod
#     @st.cache_data(ttl=5 * 60)
#     def parse_feed(feed_url: str, max_items: int = 10) -> List[Dict[str, Any]]:
#         """
#         Parse RSS/Atom feed and return items.
#         Cached 5 minutes.
#         """
#         items: List[Dict[str, Any]] = []
#         try:
#             parsed = feedparser.parse(feed_url)
#             for e in (parsed.entries or [])[:max_items]:
#                 link = getattr(e, "link", None)
#                 if not link:
#                     continue
#                 items.append(
#                     {
#                         "title": getattr(e, "title", "") or "",
#                         "url": TextUtils.normalize_url(link),
#                         "published": getattr(e, "published", None)
#                         or getattr(e, "updated", None),
#                     }
#                 )
#         except Exception:
#             pass
#         return items

#     @staticmethod
#     def collect_links(
#         selected_sources: List[Dict[str, str]], max_items_per_source: int
#     ) -> List[Dict[str, Any]]:
#         """Collect links from multiple sources."""
#         all_links: List[Dict[str, Any]] = []

#         for s in selected_sources:
#             source_name = s["source"]
#             home = s["link"].rstrip("/")

#             feed = FeedDiscoverer.discover_best_feed(home)
#             if not feed:
#                 continue

#             items = FeedDiscoverer.parse_feed(feed, max_items=max_items_per_source)
#             for it in items:
#                 all_links.append(
#                     {
#                         "source": source_name,
#                         "url": it["url"],
#                         "feed_title": it.get("title", ""),
#                         "published": it.get("published"),
#                     }
#                 )

#             time.sleep(Config.POLITE_SLEEP)

#         # Dedup based on URL
#         seen = set()
#         deduped = []
#         for row in all_links:
#             if row["url"] not in seen:
#                 seen.add(row["url"])
#                 deduped.append(row)

#         return deduped


# class ArticleExtractor:
#     """Handles article text extraction."""

#     @staticmethod
#     def extract_one(url: str) -> Dict[str, Any]:
#         """Extract article text using Trafilatura."""
#         out = {
#             "ok": False,
#             "url": url,
#             "title": "",
#             "text": "",
#             "date_published": None,
#             "error": None,
#         }
#         try:
#             downloaded = trafilatura.fetch_url(url)
#             if not downloaded:
#                 out["error"] = "fetch_url empty"
#                 return out

#             extracted = trafilatura.extract(
#                 downloaded, output_format="json", with_metadata=True
#             )
#             if not extracted:
#                 out["error"] = "extract empty"
#                 return out

#             data = json.loads(extracted)
#             out["title"] = (data.get("title") or "").strip()
#             out["text"] = (data.get("text") or "").strip()
#             out["date_published"] = data.get("date") or data.get("date_published")
#             out["ok"] = bool(out["text"])
#             return out
#         except Exception as e:
#             out["error"] = str(e)
#             return out

#     @staticmethod
#     def extract_parallel(
#         links: List[Dict[str, Any]], workers: int
#     ) -> List[Dict[str, Any]]:
#         """Extract articles concurrently."""
#         out: List[Dict[str, Any]] = []

#         with ThreadPoolExecutor(max_workers=workers) as ex:
#             future_map = {
#                 ex.submit(ArticleExtractor.extract_one, row["url"]): row
#                 for row in links
#             }

#             for fut in as_completed(future_map):
#                 meta = future_map[fut]
#                 try:
#                     art = fut.result()
#                     art["source"] = meta["source"]
#                     art["feed_published"] = meta.get("published")
#                     art["feed_title"] = meta.get("feed_title", "")
#                     out.append(art)
#                 except Exception:
#                     continue  # Should be handled inside extract_one, but safety net

#         return [a for a in out if a.get("ok")]


# class IncidentAnalyzer:
#     """Handles LLM interactions for incident analysis."""

#     def __init__(
#         self,
#         backend: str,
#         ollama_model: str,
#         groq_model: str,
#         temperature: float,
#         groq_tpm_limit: int = 6000,
#     ):
#         self.backend = backend
#         self.llm = self._build_llm(backend, ollama_model, groq_model, temperature)
#         self.text_parser = StrOutputParser()
#         self.groq_bucket = (
#             RateLimiter(groq_tpm_limit) if backend == "Groq (cloud)" else None
#         )
#         self._init_prompts_and_chains()

#     def _build_llm(
#         self, backend: str, ollama_model: str, groq_model: str, temperature: float
#     ):
#         if backend == "Ollama (local, free)":
#             return ChatOllama(model=ollama_model, temperature=temperature)
#         return ChatGroq(model=groq_model, temperature=temperature)

#     def _init_prompts_and_chains(self):
#         # Stage A: Gate
#         self.gate_prompt = ChatPromptTemplate.from_template(
#             """
# Classify the incident into exactly ONE category:
# criminality, drugs, stabbing, robbery, bombing, arson, protest, maritime, rape, air, traffic, Other.

# Return ONLY the category name.

# Text:
# {context}
# """
#         )
#         self.gate_chain = self.gate_prompt | self.llm | self.text_parser

#         # Stage B1: Impact
#         self.impact_prompt = ChatPromptTemplate.from_template(
#             """
# Classify impact level as one word: High, Medium, or Low.

# High: fatalities OR national security risk OR mass casualties
# Medium: injuries OR localized disruption, no fatalities
# Low: minor, no casualties, minimal disruption

# Return ONLY one word.

# Article:
# {context}
# """
#         )
#         self.impact_chain = self.impact_prompt | self.llm | self.text_parser

#         # Stage B2: Details
#         self.details_prompt = ChatPromptTemplate.from_template(
#             """
# You are an incident analyst.

# Return ONLY valid JSON (no markdown, no extra text) with this schema:
# {{
#   "location": {{
#     "country": string|null,
#     "city": string|null,
#     "address": string|null
#   }},
#   "summary": string
# }}

# Rules:
# - summary must be exactly 5 to 6 sentences, factual, no speculation.
# - use null for unknown location fields.

# Text:
# {context}
# """
#         )
#         self.details_chain = self.details_prompt | self.llm | self.text_parser

#     def _estimate_tokens(self, text: str) -> int:
#         return max(250, int(len(text) / 4))

#     def _invoke_with_limits(self, chain, payload: Dict[str, Any], max_retries: int = 6):
#         """Invoke chain with rate limits if using Groq."""
#         if self.backend != "Groq (cloud)":
#             return chain.invoke(payload)

#         ctx = payload.get("context", "") or ""
#         self.groq_bucket.wait_for_budget(self._estimate_tokens(ctx))

#         for attempt in range(max_retries):
#             try:
#                 return chain.invoke(payload)
#             except Exception as e:
#                 msg = str(e).lower()
#                 if "rate_limit" in msg or "429" in msg or "tpm" in msg:
#                     sleep_s = min(4.0, (0.5 * (2**attempt))) + random.uniform(
#                         0, 0.2
#                     )
#                     time.sleep(sleep_s)
#                     continue
#                 raise
#         raise RuntimeError("Groq rate-limit retries exceeded")

#     def analyze_gate(
#         self, article: Dict[str, Any], keyword_prefilter: bool
#     ) -> Dict[str, Any]:
#         """Stage A: Gate (incident type)."""
#         title = article.get("title") or article.get("feed_title") or ""
#         snippet = (article.get("text") or "")[:400]

#         if keyword_prefilter:
#             combined_text = f"{title} {snippet}"
#             if not Config.INCIDENT_KEYWORDS.search(combined_text):
#                 article["incident_type"] = "Other"
#                 return article

#         ctx = (
#             title + "\n\n" + (article.get("text") or "")[: Config.GATE_TEXT_CHARS]
#         ).strip()
#         raw = self._invoke_with_limits(self.gate_chain, {"context": ctx})
#         article["incident_type"] = TextUtils.remove_think_blocks(raw).strip()
#         return article

#     def analyze_impact(self, article: Dict[str, Any]) -> Dict[str, Any]:
#         """Stage B1: Impact classification."""
#         title = article.get("title") or article.get("feed_title") or ""
#         text = article.get("text") or ""
#         ctx = (title + "\n\n" + text[: Config.DETAILS_TEXT_CHARS]).strip()

#         raw = self._invoke_with_limits(self.impact_chain, {"context": ctx})
#         impact = TextUtils.remove_think_blocks(raw).strip()

#         impact_norm = impact.capitalize()
#         if impact_norm not in {"High", "Medium", "Low"}:
#             impact_norm = None

#         article["impact"] = impact_norm
#         return article

#     def analyze_details(self, article: Dict[str, Any]) -> Dict[str, Any]:
#         """Stage B2: Details extraction."""
#         title = article.get("title") or article.get("feed_title") or ""
#         text = article.get("text") or ""
#         ctx = (title + "\n\n" + text[: Config.DETAILS_TEXT_CHARS]).strip()

#         raw = self._invoke_with_limits(self.details_chain, {"context": ctx})
#         obj, err = TextUtils.safe_parse_json(raw)

#         if obj is None:
#             obj = {
#                 "location": {
#                     "country": None,
#                     "city": None,
#                     "address": None,
#                     "error": err,
#                 },
#                 "summary": "",
#             }

#         loc = obj.get("location") or {"country": None, "city": None, "address": None}

#         for k in ["country", "city", "address"]:
#             if k in loc and isinstance(loc[k], str) and not loc[k].strip():
#                 loc[k] = None

#         if loc.get("city") and not loc.get("country"):
#             inferred = GeoUtils.city_to_country(loc["city"])
#             if inferred:
#                 loc["country"] = inferred

#         article["country"] = loc.get("country")
#         article["city"] = loc.get("city")
#         article["address"] = loc.get("address")
#         article["location_json"] = json.dumps(loc, ensure_ascii=False)
#         article["summary"] = obj.get("summary") or ""

#         return article

#     def process_parallel(
#         self,
#         articles: List[Dict[str, Any]],
#         stage: str,
#         workers: int,
#         keyword_prefilter: bool = False,
#     ) -> List[Dict[str, Any]]:
#         """Run analysis stages in parallel."""
#         if not articles:
#             return articles

#         # Filter for stages B1/B2
#         process_list = articles
#         if stage in ["impact", "details"]:
#             process_list = [
#                 a
#                 for a in articles
#                 if (a.get("incident_type") or "").strip().lower() != "other"
#             ]

#         if not process_list:
#             return articles

#         with ThreadPoolExecutor(max_workers=workers) as ex:
#             if stage == "gate":
#                 futs = [
#                     ex.submit(self.analyze_gate, a, keyword_prefilter)
#                     for a in process_list
#                 ]
#             elif stage == "impact":
#                 futs = [ex.submit(self.analyze_impact, a) for a in process_list]
#             elif stage == "details":
#                 futs = [ex.submit(self.analyze_details, a) for a in process_list]
#             else:
#                 return articles

#             for _ in as_completed(futs):
#                 pass
        
#         return articles


# # ==============================
# # MAIN APPLICATION
# # ==============================


# def main():
#     """Main Streamlit application entry point."""
#     st.set_page_config(page_title="⚡ Incident Collector (Refactored)", layout="wide")
#     st.title("⚡ Incident Intelligence Collector")

#     # Sidebar / Intro
#     st.markdown(
#         """
#     ### Overview
#     - **Backend**: LLMs models (cloud or Local).
#     - **Pipeline**: Feed Discovery -> Extraction -> Classify -> Impact -> Details.
#     """
#     )

#     # 1. Configuration Inputs
#     # -----------------------
#     st.subheader("1. Source Configuration")
#     default_csv = r"C:\\Yasser-SSD\\Langchain_Projects_01012026\\RFP_Proposal_Generator\\list_of_news_sources.csv"
#     csv_path = st.text_input("Path to Sources CSV", value=default_csv)

#     if not os.path.exists(csv_path):
#         st.error(f"CSV not found: {csv_path}")
#         st.stop()

#     sources_df = pd.read_csv(csv_path)
#     # Norm columns
#     sources_df.columns = [c.lower() for c in sources_df.columns]
#     if not {"source", "link"}.issubset(set(sources_df.columns)):
#         st.error("CSV must contain columns: source, link")
#         st.stop()

#     q = st.text_input("Filter sources", value="").strip().lower()
#     filtered = sources_df
#     if q:
#         filtered = filtered[
#             filtered["source"].astype(str).str.lower().str.contains(q)
#             | filtered["link"].astype(str).str.lower().str.contains(q)
#         ]

#     labels = [f"{row['source']} — {row['link']}" for _, row in filtered.iterrows()]
#     selected_labels = st.multiselect(
#         "Select sources", options=labels, default=labels[:5] if labels else []
#     )

#     selected_sources = []
#     for lbl in selected_labels:
#         name, link = lbl.split(" — ", 1)
#         selected_sources.append({"source": name.strip(), "link": link.strip()})

#     # 2. Pipeline Settings
#     # --------------------
#     st.subheader("2. Pipeline Settings")
#     colA, colB, colC, colD = st.columns(4)
#     with colA:
#         max_items = st.number_input("Items/Source", 1, 500, 10, key="max_items")
#     with colB:
#         max_total = st.number_input("Total Cap", 1, 500, 10, key="max_total")
#     with colC:
#         do_keyword = st.checkbox("Keyword Pre-filter", value=True)
#     with colD:
#         extract_workers = st.number_input(
#             "Extract Workers", 1, 20, Config.DEFAULT_EXTRACT_WORKERS
#         )

#     # 3. LLM Settings
#     # ---------------
#     st.subheader("3. Model Settings")
#     backend = st.selectbox(
#         "Backend", ["Ollama (local, free)", "Groq (cloud)"], index=0
#     )

#     c1, c2, c3 = st.columns(3)
#     with c1:
#         temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
#     with c2:
#         ollama_mod = st.text_input("Ollama Model", "qwen2.5:7b-instruct")
#     with c3:
#         groq_mod = st.text_input("Groq Model", "qwen/qwen3-32b")

#     # Advanced Groq settings
#     groq_bucket_limit = 6000
#     llm_workers_def = (
#         Config.DEFAULT_LLM_WORKERS_OLLAMA
#         if backend.startswith("Ollama")
#         else Config.DEFAULT_LLM_WORKERS_GROQ
#     )
    
#     c4, c5 = st.columns(2)
#     with c4:
#         llm_workers = st.number_input(
#             "LLM Workers", 1, 10, llm_workers_def
#         )
#     with c5:
#         if backend.startswith("Groq"):
#             groq_bucket_limit = st.number_input("Groq TPM Limit", 1000, 20000, 6000, 500)

#     # 4. Execution
#     # ------------
#     if st.button("🚀 Run Pipeline"):
#         if not selected_sources:
#             st.warning("Please select at least one source.")
#             st.stop()

#         start_time = time.time()

#         # Initialize Analyzer
#         analyzer = IncidentAnalyzer(
#             backend=backend,
#             ollama_model=ollama_mod.strip(),
#             groq_model=groq_mod.strip(),
#             temperature=float(temp),
#             groq_tpm_limit=int(groq_bucket_limit),
#         )

#         # Step 1: Collect Links
#         with st.spinner("Collecting links..."):
#             links = FeedDiscoverer.collect_links(selected_sources, int(max_items))
#             links = links[: int(max_total)]
#         st.write(f"Collected URLs: **{len(links)}**")

#         # Step 2: Extract
#         with st.spinner(f"Extracting articles ({extract_workers} workers)..."):
#             articles = ArticleExtractor.extract_parallel(links, int(extract_workers))
#         st.write(f"Extracted articles: **{len(articles)}**")

#         # Step 3: Gate
#         with st.spinner("Stage A: Gate (Incident Classification)..."):
#             articles = analyzer.process_parallel(
#                 articles, "gate", int(llm_workers), keyword_prefilter=do_keyword
#             )
        
#         incidents = [
#             a for a in articles 
#             if (a.get("incident_type") or "").strip().lower() != "other"
#         ]
#         st.write(f"Incident candidates: **{len(incidents)}**")

#         # Step 4: Impact
#         with st.spinner("Stage B1: Impact Assessment..."):
#             articles = analyzer.process_parallel(articles, "impact", int(llm_workers))

#         # Step 5: Details
#         with st.spinner("Stage B2: Details Extraction..."):
#             articles = analyzer.process_parallel(articles, "details", int(llm_workers))

#         # Final Results
#         results = [
#             a
#             for a in articles
#             if (a.get("incident_type") or "").strip().lower() != "other"
#         ]

#         st.success(f"✅ Finished in {time.time() - start_time:.1f}s")
        
#         if not results:
#             st.warning("No incidents found.")
#             st.stop()

#         # Prepare DataFrame
#         results_clean = []
#         for a in results:
#             item = {
#                 "source": a.get("source"),
#                 "url": a.get("url"),
#                 "title": a.get("title") or a.get("feed_title"),
#                 "date": a.get("date_published") or a.get("feed_published"),
#                 "type": a.get("incident_type"),
#                 "impact": a.get("impact"),
#                 "country": a.get("country"),
#                 "city": a.get("city"),
#                 "summary": a.get("summary"),
#             }
#             results_clean.append(item)

#         df = pd.DataFrame(results_clean)

#         st.dataframe(df, use_container_width=True)
        
#         # Download buttons
#         c_csv, c_json = st.columns(2)
#         with c_csv:
#             st.download_button(
#                 "Download CSV",
#                 df.to_csv(index=False).encode("utf-8"),
#                 "incidents.csv",
#                 "text/csv",
#             )
#         with c_json:
#             st.download_button(
#                 "Download JSON",
#                 json.dumps(results_clean, indent=2, ensure_ascii=False).encode("utf-8"),
#                 "incidents.json",
#                 "application/json",
#             )


# if __name__ == "__main__":
#     main()


