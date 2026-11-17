"""End-to-end radar scan: discover -> extract -> LLM triage -> LLM analysis -> geocode -> store."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from news_radar.analyzer import IncidentAnalyzer
from news_radar.config import ScanConfig
from news_radar.discovery import NewsDiscovery, canonical_url, filter_unseen
from news_radar.geo import Geocoder
from news_radar.llm import LLMClient, build_llm
from news_radar.models import Incident
from news_radar.store import RadarStore

logger = logging.getLogger(__name__)

# progress(stage, message, fraction 0..1)
ProgressFn = Callable[[str, str, float], None]


@dataclass
class ScanResult:
    incidents: List[Incident] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


def _noop(stage: str, message: str, fraction: float) -> None:
    logger.info("[%s] %s (%.0f%%)", stage, message, fraction * 100)


class RadarPipeline:
    def __init__(
        self,
        config: ScanConfig,
        store: Optional[RadarStore] = None,
        llm: Optional[LLMClient] = None,
        discovery: Optional[NewsDiscovery] = None,
        geocoder: Optional[Geocoder] = None,
    ):
        self.cfg = config
        self.store = store or RadarStore(config.db_path)
        self.llm = llm or build_llm(config.llm)
        self.discovery = discovery or NewsDiscovery(config)
        self.geocoder = geocoder or Geocoder(config.geo, cache=self.store)
        self.analyzer = IncidentAnalyzer(self.llm, config.llm, cache=self.store)

    def run(self, progress: ProgressFn = _noop) -> ScanResult:
        t0 = time.time()
        res = ScanResult()
        st = res.stats

        # 1. Discover URLs ------------------------------------------------------
        progress("discover", "Searching global news…", 0.02)
        url_via = self.discovery.discover_urls()
        ds = self.discovery.stats
        st.update(discovery_tasks=ds.tasks, discovery_failed=ds.failed_tasks,
                  urls_found=ds.urls_found, urls_unique=ds.urls_unique, urls_undecoded=ds.urls_undecoded)
        if self.cfg.skip_seen:
            before = len(url_via)
            url_via = filter_unseen(url_via, self.store.is_seen)
            st["urls_skipped_seen"] = before - len(url_via)
        url_via = url_via[: self.cfg.max_articles]
        progress("discover", f"{len(url_via)} new article URLs", 0.15)

        if not url_via:
            res.errors = list(ds.errors)
            st["elapsed_s"] = round(time.time() - t0, 1)
            return res

        # 2. Extract -------------------------------------------------------------
        progress("extract", f"Extracting {len(url_via)} articles…", 0.18)
        articles = self.discovery.extract(url_via)
        st.update(extracted=len(articles), duplicates_by_title=ds.duplicates_by_title)
        extracted_canon = {canonical_url(a.url) for a in articles}
        failed = [(u, "failed") for u, _ in url_via if canonical_url(u) not in extracted_canon]
        progress("extract", f"{len(articles)} articles extracted", 0.40)

        # 3. Triage --------------------------------------------------------------
        def tprog(done: int, total: int) -> None:
            progress("triage", f"LLM triage {done}/{total}", 0.40 + 0.20 * done / max(total, 1))

        labelled = self.analyzer.triage(articles, progress=tprog)
        incident_articles = [a for a, t in labelled if t.is_incident and t.confidence >= self.cfg.min_confidence]
        st.update(triaged=len(labelled), incidents_detected=len(incident_articles))
        progress("triage", f"{len(incident_articles)} incidents out of {len(labelled)} articles", 0.60)

        # 4. Analysis ------------------------------------------------------------
        def aprog(done: int, total: int) -> None:
            progress("analyze", f"Categorising & summarising {done}/{total}", 0.60 + 0.30 * done / max(total, 1))

        incidents = self.analyzer.analyze_many(incident_articles, progress=aprog)
        st["analyzed"] = len(incidents)

        # 5. Geocode -------------------------------------------------------------
        progress("geocode", "Placing incidents on the map…", 0.92)
        for inc in incidents:
            try:
                self.geocoder.geocode(inc.location)
            except Exception as e:  # never lose an incident because of geocoding
                res.errors.append(f"geocode failed for {inc.url}: {e}")
        st["mapped"] = sum(1 for i in incidents if i.location.has_coords())

        # 6. Persist -------------------------------------------------------------
        self.store.save_incidents(incidents)
        # Mark as seen only what reached a final state; failed analyses and unlabeled
        # articles are retried next run (their LLM results are cached where available).
        disc_by_canon = {canonical_url(u): u for u, _ in url_via}
        analysed = {i.url for i in incidents}
        seen = list(failed)
        for a, t in labelled:
            if a.url in analysed:
                label = "incident"
            elif not t.is_incident:
                label = "non-incident"
            else:
                continue
            seen.append((a.url, label))
            discovered = disc_by_canon.get(canonical_url(a.url))
            if discovered and discovered != a.url:
                seen.append((discovered, label))
        self.store.mark_seen(seen)

        res.incidents = sorted(incidents, key=lambda i: ["Critical", "High", "Medium", "Low"].index(i.severity))
        res.errors = list(ds.errors) + self.analyzer.errors + res.errors
        st["elapsed_s"] = round(time.time() - t0, 1)
        progress("done", f"{len(incidents)} incidents mapped in {st['elapsed_s']}s", 1.0)
        return res
