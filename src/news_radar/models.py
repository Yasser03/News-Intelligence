"""Typed data models shared across the pipeline."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from pydantic import BaseModel, Field, field_validator

_TRACKING_PARAMS = re.compile(r"^(utm_|fbclid|gclid|ocid|cmpid|ref$|ref_)", re.I)


def canonical_url(url: str) -> str:
    """Normalise a URL for deduplication (drop fragment & tracking params, lowercase host)."""
    try:
        p = urlparse(url.strip())
        query = urlencode([(k, v) for k, v in parse_qsl(p.query) if not _TRACKING_PARAMS.match(k)])
        path = p.path.rstrip("/") or "/"
        return urlunparse((p.scheme.lower(), p.netloc.lower(), path, "", query, ""))
    except Exception:
        return url


def url_id(url: str) -> str:
    return hashlib.sha256(canonical_url(url).encode("utf-8")).hexdigest()[:24]


class Article(BaseModel):
    url: str
    title: str = ""
    text: str = ""
    description: str = ""
    source: str = ""
    author: str = ""
    published: str = ""        # ISO date or datetime string as extracted
    language: str = ""
    discovered_via: str = ""   # e.g. "google-news:US:explosion" or "source:https://..."

    @property
    def id(self) -> str:
        return url_id(self.url)

    def snippet(self, n: int = 350) -> str:
        body = self.description or self.text
        return " ".join(body.split())[:n]


class Location(BaseModel):
    country: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    place: Optional[str] = None   # street, landmark, facility
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    precision: Optional[str] = None  # place | city | region | country | llm-estimate

    @field_validator("country", "region", "city", "place", mode="before")
    @classmethod
    def _blank_to_none(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip().lower() in {"", "null", "none", "unknown", "n/a"}:
            return None
        return v

    @field_validator("latitude", "longitude", mode="before")
    @classmethod
    def _coerce_float(cls, v: Any) -> Any:
        if v in (None, ""):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def has_coords(self) -> bool:
        return (
            self.latitude is not None
            and self.longitude is not None
            and -90 <= self.latitude <= 90
            and -180 <= self.longitude <= 180
        )

    def label(self) -> str:
        return ", ".join(p for p in [self.place, self.city, self.region, self.country] if p)


class TriageResult(BaseModel):
    id: int
    is_incident: bool
    confidence: float = 0.0


class Incident(BaseModel):
    id: str
    url: str
    source: str = ""
    article_title: str = ""
    headline: str = ""
    category: str
    subcategory: str
    severity: str
    summary: str = ""
    event_date: Optional[str] = None
    published: str = ""
    location: Location = Field(default_factory=Location)
    killed: Optional[int] = None
    injured: Optional[int] = None
    actors: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    language: str = ""
    discovered_via: str = ""
    scanned_at: str = ""
    model: str = ""

    def flat(self) -> Dict[str, Any]:
        """Flat dict for tables / CSV."""
        d = self.model_dump(exclude={"location"})
        loc = self.location
        d.update(
            country=loc.country,
            region=loc.region,
            city=loc.city,
            place=loc.place,
            latitude=loc.latitude,
            longitude=loc.longitude,
            geo_precision=loc.precision,
            actors=", ".join(self.actors),
        )
        return d
