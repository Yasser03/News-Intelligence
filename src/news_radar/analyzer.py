"""LLM stages: (1) batched Incident / Non-Incident triage, (2) per-incident categorisation & summary."""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from news_radar.config import LLMConfig
from news_radar.llm import LLMClient, parse_json
from news_radar.models import Article, Incident, Location, TriageResult
from news_radar.taxonomy import SEVERITY_GUIDE, normalize_category, normalize_severity, taxonomy_prompt_block

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v1"


class JsonCache(Protocol):
    def get(self, key: str) -> Optional[Any]: ...
    def put(self, key: str, value: Any) -> None: ...


TRIAGE_PROMPT = """You are an OSINT analyst screening world news for a security incident radar.

For EACH numbered item decide if it reports a specific, real-world security or safety INCIDENT:
a concrete event that has happened or is unfolding at an identifiable place, such as armed conflict,
terrorism, violent crime, organized crime, protests/riots, transport accidents, natural disasters,
explosions/fires/industrial accidents, disease outbreaks, or cyberattacks.

Label "Non-Incident" for: opinion, analysis or explainers, politics/diplomacy without a violent or
hazardous event, business/markets, sports, entertainment, weather forecasts, anniversaries,
court proceedings about old events, policy announcements, and general statistics or trends.

Return ONLY a JSON object, no prose:
{{"results": [{{"id": <item id>, "label": "Incident" | "Non-Incident", "confidence": <0.0-1.0>}}, ...]}}
Include every id exactly once.

Items:
{items}
"""

ANALYSIS_PROMPT = """You are an OSINT incident analyst. Analyse the news article below and return ONLY a JSON object.

Taxonomy (choose exactly one category and one of ITS sub-categories):
{taxonomy}

Severity levels: {severity_guide}

JSON schema:
{{
  "category": string,
  "subcategory": string,
  "severity": "Critical" | "High" | "Medium" | "Low",
  "headline": string,               // neutral 8-14 word headline of the incident
  "summary": string,                // 3-5 factual sentences: what, where, when, who, impact. No speculation.
  "event_date": "YYYY-MM-DD" | null,
  "location": {{
    "country": string | null,       // English country name
    "region": string | null,        // state / province
    "city": string | null,
    "place": string | null,         // street, landmark or facility if stated
    "latitude": number | null,      // your best estimate for the most specific location
    "longitude": number | null
  }},
  "killed": integer | null,
  "injured": integer | null,
  "actors": [string],               // groups, organisations or agencies involved (max 5)
  "confidence": number              // 0.0-1.0 that this is a real incident correctly classified
}}

If several incidents are mentioned, describe the main one. Write the output in English even if the article is not.

Published: {published}
Source: {source}
Title: {title}

Article:
{text}
"""


def _cache_key(kind: str, model: str, *parts: str) -> str:
    h = hashlib.sha256()
    for p in (kind, PROMPT_VERSION, model, *parts):
        h.update(p.encode("utf-8", "ignore"))
        h.update(b"\x1f")
    return f"{kind}:{h.hexdigest()}"


def _to_int(v: Any) -> Optional[int]:
    try:
        if v is None or v == "":
            return None
        n = int(float(v))
        return n if n >= 0 else None
    except (TypeError, ValueError):
        return None


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


class IncidentAnalyzer:
    def __init__(self, llm: LLMClient, cfg: LLMConfig, cache: Optional[JsonCache] = None,
                 triage_llm: Optional[LLMClient] = None):
        self.llm = llm
        self.triage_llm = triage_llm or llm
        self.cfg = cfg
        self.cache = cache
        self.errors: List[str] = []

    # ------------------------------------------------------------------ triage
    def _triage_batch(self, batch: Sequence[Article]) -> Dict[int, TriageResult]:
        lines = []
        for i, a in enumerate(batch, start=1):
            lines.append(f"[{i}] {a.title.strip()} — {a.snippet(300)}")
        prompt = TRIAGE_PROMPT.format(items="\n".join(lines))
        data = parse_json(self.triage_llm.complete(prompt))
        rows = data.get("results", data) if isinstance(data, dict) else data
        out: Dict[int, TriageResult] = {}
        for r in rows if isinstance(rows, list) else []:
            if not isinstance(r, dict):
                continue
            try:
                idx = int(r.get("id"))
            except (TypeError, ValueError):
                continue
            label = str(r.get("label", "")).strip().lower().replace("_", "-")
            out[idx] = TriageResult(
                id=idx,
                is_incident=label in {"incident", "yes", "true"},
                confidence=_to_float(r.get("confidence"), 0.5),
            )
        return out

    def triage(
        self,
        articles: Sequence[Article],
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[Tuple[Article, TriageResult]]:
        """Label every article. Results come back in input order."""
        results: Dict[str, TriageResult] = {}
        todo: List[Article] = []
        for a in articles:
            key = _cache_key("triage", self.triage_llm.model_name, a.url, a.title)
            cached = self.cache.get(key) if self.cache else None
            if cached:
                results[a.url] = TriageResult(**cached)
            else:
                todo.append(a)

        size = max(1, self.cfg.triage_batch_size)
        batches = [todo[i : i + size] for i in range(0, len(todo), size)]
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, self.cfg.workers)) as ex:
            futs = {ex.submit(self._triage_batch, b): b for b in batches}
            for fut in as_completed(futs):
                batch = futs[fut]
                try:
                    labels = fut.result()
                except Exception as e:
                    self.errors.append(f"triage batch failed: {e}")
                    logger.warning("Triage batch failed: %s", e)
                    labels = {}
                for i, a in enumerate(batch, start=1):
                    tr = labels.get(i)
                    if tr is None:
                        continue  # unlabeled -> retried next run (not cached)
                    results[a.url] = tr
                    if self.cache:
                        self.cache.put(_cache_key("triage", self.triage_llm.model_name, a.url, a.title), tr.model_dump())
                done += len(batch)
                if progress:
                    progress(done, len(todo))

        return [(a, results[a.url]) for a in articles if a.url in results]

    # ---------------------------------------------------------------- analysis
    def analyze(self, article: Article) -> Optional[Incident]:
        key = _cache_key("analysis", self.llm.model_name, article.url, article.title)
        data = self.cache.get(key) if self.cache else None
        if data is None:
            body = (article.text or article.description or "")[: self.cfg.max_chars]
            prompt = ANALYSIS_PROMPT.format(
                taxonomy=taxonomy_prompt_block(),
                severity_guide=SEVERITY_GUIDE,
                published=article.published or "unknown",
                source=article.source or "unknown",
                title=article.title,
                text=body or "(no body text available)",
            )
            data = parse_json(self.llm.complete(prompt))
            if not isinstance(data, dict):
                raise ValueError("analysis output is not a JSON object")
            if self.cache:
                self.cache.put(key, data)
        return self._build_incident(article, data)

    def _build_incident(self, article: Article, d: Dict[str, Any]) -> Incident:
        category, subcategory = normalize_category(d.get("category"), d.get("subcategory"))
        loc_raw = d.get("location") if isinstance(d.get("location"), dict) else {}
        location = Location(**{k: loc_raw.get(k) for k in ("country", "region", "city", "place", "latitude", "longitude")})
        if location.has_coords():
            location.precision = "llm-estimate"
        else:
            location.latitude = location.longitude = None
        actors = d.get("actors") if isinstance(d.get("actors"), list) else []
        return Incident(
            id=article.id,
            url=article.url,
            source=article.source,
            article_title=article.title,
            headline=str(d.get("headline") or article.title).strip(),
            category=category,
            subcategory=subcategory,
            severity=normalize_severity(d.get("severity")),
            summary=str(d.get("summary") or "").strip(),
            event_date=(str(d["event_date"])[:10] if d.get("event_date") else None),
            published=article.published,
            location=location,
            killed=_to_int(d.get("killed")),
            injured=_to_int(d.get("injured")),
            actors=[str(x).strip() for x in actors if str(x).strip()][:5],
            confidence=_to_float(d.get("confidence"), 0.5),
            language=article.language,
            discovered_via=article.discovered_via,
            scanned_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            model=self.llm.model_name,
        )

    def analyze_many(
        self,
        articles: Sequence[Article],
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[Incident]:
        incidents: List[Incident] = []
        with ThreadPoolExecutor(max_workers=max(1, self.cfg.workers)) as ex:
            futs = {ex.submit(self.analyze, a): a for a in articles}
            for n, fut in enumerate(as_completed(futs), start=1):
                a = futs[fut]
                try:
                    inc = fut.result()
                    if inc:
                        incidents.append(inc)
                except Exception as e:
                    self.errors.append(f"analysis failed for {a.url}: {e}")
                    logger.warning("Analysis failed for %s: %s", a.url, e)
                if progress:
                    progress(n, len(articles))
        return incidents


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)
