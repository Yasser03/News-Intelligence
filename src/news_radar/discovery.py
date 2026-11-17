"""News discovery and extraction, powered by pipescraper.

* Global mode (default, no source list needed): sweeps Google News country editions with
  incident-oriented search terms via ``FetchGoogleNews``.
* Custom sources (optional): crawls home/section pages for article links via ``FetchLinks``.
* Extraction: full text + metadata for every URL via ``ExtractArticles``.

The three network operations are injectable so the pipeline can be tested offline.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from news_radar.config import ScanConfig
from news_radar.models import Article, canonical_url

logger = logging.getLogger(__name__)

# (country, period, max_results, query or None) -> urls
GoogleFetchFn = Callable[[str, str, int, Optional[str]], List[str]]
# (source_url, max_links, respect_robots) -> urls
LinksFetchFn = Callable[[str, int, bool], List[str]]
# (urls, workers, timeout) -> list of objects with ScrapedArticle-like attributes
ExtractFn = Callable[[List[str], int, int], List[Any]]

def title_key(title: str) -> str:
    """Key for near-duplicate headlines across outlets ("X - Reuters" vs "X | BBC")."""
    t = re.split(r"\s[-|–—]\s(?=[^-|–—]+$)", title or "")[0]
    words = re.sub(r"[^a-z0-9 ]+", " ", t.lower()).split()
    return " ".join(words[:12])


# ---------------------------------------------------------------------------
# Default (live) implementations using pipescraper
# ---------------------------------------------------------------------------

class GoogleNewsUnavailable(RuntimeError):
    pass


def pipescraper_google_news(country: str, period: str, max_results: int, query: Optional[str]) -> List[str]:
    from pipescraper import Deduplicate, FetchGoogleNews

    try:
        return (
            FetchGoogleNews(country=country, period=period, max_results=max_results, search=query)
            >> Deduplicate()
        )
    except UnboundLocalError as e:
        # pipescraper<=0.3.0 raises this when neither gnews nor newspaper4k's Google source is usable.
        raise GoogleNewsUnavailable(
            "Google News backend unavailable - install it with: pip install gnews 'newspaper4k[gnews]'"
        ) from e


def pipescraper_links(source_url: str, max_links: int, respect_robots: bool) -> List[str]:
    from pipescraper import FetchLinks

    return source_url >> FetchLinks(max_links=max_links, respect_robots=respect_robots)


def pipescraper_extract(urls: List[str], workers: int, timeout: int) -> List[Any]:
    from pipescraper import ExtractArticles

    if not urls:
        return []
    return urls >> ExtractArticles(
        workers=workers,
        timeout=timeout,
        extract_time=True,
        user_agent="Mozilla/5.0 (compatible; osint-news-radar/2.0)",
    )


# ---------------------------------------------------------------------------
# Discovery orchestration
# ---------------------------------------------------------------------------

@dataclass
class DiscoveryStats:
    tasks: int = 0
    failed_tasks: int = 0
    urls_found: int = 0
    urls_unique: int = 0
    urls_undecoded: int = 0
    urls_skipped_seen: int = 0
    extracted: int = 0
    duplicates_by_title: int = 0
    errors: List[str] = field(default_factory=list)


class NewsDiscovery:
    def __init__(
        self,
        config: ScanConfig,
        google_fetch: GoogleFetchFn = pipescraper_google_news,
        links_fetch: LinksFetchFn = pipescraper_links,
        extract: ExtractFn = pipescraper_extract,
        discovery_workers: int = 6,
    ):
        self.cfg = config
        self.google_fetch = google_fetch
        self.links_fetch = links_fetch
        self.extract_fn = extract
        self.discovery_workers = discovery_workers
        self.stats = DiscoveryStats()

    # -- step 1: URLs ------------------------------------------------------
    def _google_tasks(self) -> List[Tuple[str, Optional[str]]]:
        if not self.cfg.global_scan:
            return []
        focus = self.cfg.focus.strip()
        queries: List[Optional[str]] = [f"{q} {focus}".strip() for q in self.cfg.queries if q.strip()]
        if self.cfg.include_top_news or not queries:
            queries.append(focus or None)
        return [(c, q) for c in self.cfg.countries for q in queries]

    def discover_urls(self) -> List[Tuple[str, str]]:
        """Return ``[(url, discovered_via), ...]`` deduplicated, in discovery order."""
        jobs: List[Tuple[str, Callable[[], List[str]]]] = []
        for country, query in self._google_tasks():
            via = f"google-news:{country}:{query or 'top'}"
            jobs.append((via, lambda c=country, q=query: self.google_fetch(c, self.cfg.period, self.cfg.results_per_query, q)))
        for src in self.cfg.sources:
            src = src.strip()
            if src:
                jobs.append((f"source:{src}", lambda s=src: self.links_fetch(s, self.cfg.links_per_source, self.cfg.respect_robots)))

        self.stats.tasks = len(jobs)
        found: List[Tuple[str, str]] = []
        with ThreadPoolExecutor(max_workers=max(1, self.discovery_workers)) as ex:
            futures = {ex.submit(fn): via for via, fn in jobs}
            for fut in as_completed(futures):
                via = futures[fut]
                try:
                    urls = fut.result() or []
                except Exception as e:  # one bad source must not stop the scan
                    self.stats.failed_tasks += 1
                    self.stats.errors.append(f"{via}: {e}")
                    logger.warning("Discovery task failed %s: %s", via, e)
                    continue
                found.extend((u, via) for u in urls if isinstance(u, str) and u.startswith("http"))

        self.stats.urls_found = len(found)
        seen, unique = set(), []
        for url, via in found:
            key = canonical_url(url)
            if "news.google.com/" in key:  # Google redirect that could not be decoded = unusable
                self.stats.urls_undecoded += 1
                continue
            if key in seen:
                continue
            seen.add(key)
            unique.append((url, via))
        self.stats.urls_unique = len(unique)
        return unique

    # -- step 2: articles --------------------------------------------------
    def extract(self, url_via: Sequence[Tuple[str, str]]) -> List[Article]:
        via_by_url = {canonical_url(u): v for u, v in url_via}
        urls = [u for u, _ in url_via]
        try:
            scraped = self.extract_fn(urls, self.cfg.extract_workers, self.cfg.request_timeout) or []
        except Exception as e:
            self.stats.errors.append(f"extract: {e}")
            logger.exception("Extraction failed")
            scraped = []

        articles: List[Article] = []
        titles = set()
        for s in scraped:
            art = _to_article(s)
            if art is None:
                continue
            # The page's canonical URL can differ from the discovered one; fall back to a marker.
            art.discovered_via = via_by_url.get(canonical_url(art.url), "extracted")
            tk = title_key(art.title)
            if tk and tk in titles:
                self.stats.duplicates_by_title += 1
                continue
            titles.add(tk)
            articles.append(art)
        self.stats.extracted = len(articles)
        return articles


def _to_article(s: Any) -> Optional[Article]:
    """Convert a pipescraper ScrapedArticle (or dict) to our Article model."""
    get = (lambda k: s.get(k, "")) if isinstance(s, dict) else (lambda k: getattr(s, k, "") or "")
    title, text = str(get("title") or "").strip(), str(get("text") or "").strip()
    if not get("url") or (not title and not text):
        return None
    date, time_ = str(get("date_published") or ""), str(get("time_published") or "")
    return Article(
        url=str(get("url")),
        title=title,
        text=text,
        description=str(get("description") or "").strip(),
        source=str(get("source") or urlparse(str(get("url"))).netloc),
        author=str(get("author") or ""),
        published=f"{date} {time_}".strip() if date else "",
        language=str(get("language") or ""),
    )


def filter_unseen(url_via: Iterable[Tuple[str, str]], is_seen: Callable[[str], bool]) -> List[Tuple[str, str]]:
    return [(u, v) for u, v in url_via if not is_seen(u)]
