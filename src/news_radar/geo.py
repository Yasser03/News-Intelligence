"""Geocoding: offline gazetteer first, optional Nominatim (cached, rate-limited), LLM estimate fallback.

Resolution order for each incident location:
  1. Nominatim for a specific place/street (only if enabled)       -> precision "place"
  2. Offline GeoNames gazetteer city match (country-aware)          -> precision "city"
  3. Nominatim for city / region (only if enabled)                  -> precision "city" / "region"
  4. The LLM's own coordinate estimate                              -> precision "llm-estimate"
  5. Country capital from the gazetteer                             -> precision "country"
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Dict, List, Optional, Protocol, Tuple

from news_radar.config import GeoConfig
from news_radar.models import Location

logger = logging.getLogger(__name__)

_COUNTRY_ALIASES = {
    "usa": "US", "us": "US", "u s": "US", "united states of america": "US", "america": "US",
    "uk": "GB", "u k": "GB", "britain": "GB", "great britain": "GB", "england": "GB",
    "scotland": "GB", "wales": "GB", "northern ireland": "GB",
    "russia": "RU", "russian federation": "RU", "south korea": "KR", "north korea": "KP",
    "iran": "IR", "syria": "SY", "vietnam": "VN", "laos": "LA", "bolivia": "BO", "venezuela": "VE",
    "tanzania": "TZ", "moldova": "MD", "czechia": "CZ", "czech republic": "CZ",
    "dr congo": "CD", "drc": "CD", "democratic republic of the congo": "CD",
    "republic of the congo": "CG", "congo brazzaville": "CG", "ivory coast": "CI", "cote d ivoire": "CI",
    "palestine": "PS", "gaza": "PS", "west bank": "PS", "taiwan": "TW", "turkey": "TR", "turkiye": "TR",
    "uae": "AE", "emirates": "AE", "burma": "MM", "myanmar": "MM", "eswatini": "SZ", "swaziland": "SZ",
    "east timor": "TL", "cape verde": "CV", "the gambia": "GM", "gambia": "GM", "the bahamas": "BS",
    "vatican": "VA", "macedonia": "MK", "north macedonia": "MK", "micronesia": "FM",
}


def _norm(s: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


class Gazetteer:
    """Offline city / country index built from geonamescache (cities with population > 15k)."""

    _lock = threading.Lock()
    _instance: Optional["Gazetteer"] = None

    def __init__(self) -> None:
        import geonamescache

        gc = geonamescache.GeonamesCache()
        self.countries: Dict[str, dict] = gc.get_countries()
        self.country_by_name: Dict[str, str] = {}
        for code, c in self.countries.items():
            self.country_by_name[_norm(c["name"])] = code
            self.country_by_name[_norm(c.get("iso3"))] = code
            self.country_by_name[code.lower()] = code
        self.country_by_name.update(_COUNTRY_ALIASES)

        self.cities: Dict[str, List[dict]] = {}
        for c in gc.get_cities().values():
            names = {_norm(c.get("name"))}
            # include alternate names that are short ASCII variants (e.g. "Kyiv"/"Kiev")
            for alt in c.get("alternatenames") or []:
                if alt and alt.isascii() and len(alt) <= 40:
                    names.add(_norm(alt))
            for n in names:
                if n:
                    self.cities.setdefault(n, []).append(c)
        for lst in self.cities.values():
            lst.sort(key=lambda x: x.get("population", 0), reverse=True)

    @classmethod
    def get(cls) -> "Gazetteer":
        with cls._lock:
            if cls._instance is None:
                cls._instance = Gazetteer()
            return cls._instance

    def country_code(self, country: Optional[str]) -> Optional[str]:
        return self.country_by_name.get(_norm(country)) if country else None

    def country_name(self, code: Optional[str]) -> Optional[str]:
        return self.countries.get(code or "", {}).get("name")

    def city(self, city: Optional[str], country: Optional[str] = None) -> Optional[Tuple[float, float, str]]:
        """Return ``(lat, lon, country_code)`` of the most populous matching city."""
        candidates = self.cities.get(_norm(city)) if city else None
        if not candidates:
            return None
        code = self.country_code(country)
        if code:
            candidates = [c for c in candidates if c.get("countrycode") == code]
            if not candidates:
                return None
        best = candidates[0]
        return float(best["latitude"]), float(best["longitude"]), best.get("countrycode", "")

    def country_point(self, country: Optional[str]) -> Optional[Tuple[float, float]]:
        code = self.country_code(country)
        if not code:
            return None
        capital = self.countries[code].get("capital")
        hit = self.city(capital, self.countries[code]["name"]) if capital else None
        if hit:
            return hit[0], hit[1]
        # fall back to the most populous city of that country
        best = None
        for lst in self.cities.values():
            for c in lst:
                if c.get("countrycode") == code and (best is None or c.get("population", 0) > best.get("population", 0)):
                    best = c
        return (float(best["latitude"]), float(best["longitude"])) if best else None


class GeoCache(Protocol):
    def get_geo(self, query: str) -> Optional[Tuple[Optional[float], Optional[float]]]: ...
    def put_geo(self, query: str, lat: Optional[float], lon: Optional[float]) -> None: ...


class NominatimClient:
    """Serialized, rate-limited Nominatim lookups with an optional persistent cache (misses cached too)."""

    def __init__(self, cfg: GeoConfig, cache: Optional[GeoCache] = None):
        from geopy.geocoders import Nominatim

        self._geo = Nominatim(user_agent=cfg.user_agent, timeout=10)
        self._delay = cfg.min_delay_seconds
        self._cache = cache
        self._lock = threading.Lock()
        self._last = 0.0
        self._mem: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    def lookup(self, query: str) -> Optional[Tuple[float, float]]:
        q = query.strip()
        if not q:
            return None
        hit = self._mem.get(q) or (self._cache.get_geo(q) if self._cache else None)
        if hit is not None:
            return (hit[0], hit[1]) if hit[0] is not None else None
        with self._lock:
            wait = self._delay - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                loc = self._geo.geocode(q)
            except Exception as e:
                logger.warning("Nominatim failed for %r: %s", q, e)
                return None  # transient: do not cache
            finally:
                self._last = time.monotonic()
        result = (loc.latitude, loc.longitude) if loc else (None, None)
        self._mem[q] = result
        if self._cache:
            self._cache.put_geo(q, *result)
        return (result[0], result[1]) if result[0] is not None else None


class Geocoder:
    def __init__(self, cfg: GeoConfig, cache: Optional[GeoCache] = None, nominatim: Optional[NominatimClient] = None):
        self.cfg = cfg
        self.gaz = Gazetteer.get()
        self.nominatim = nominatim or (NominatimClient(cfg, cache) if cfg.use_nominatim else None)

    def geocode(self, loc: Location) -> Location:
        llm_lat, llm_lon = (loc.latitude, loc.longitude) if loc.has_coords() else (None, None)

        # Standardise the country name when we recognise it.
        code = self.gaz.country_code(loc.country)
        if code:
            loc.country = self.gaz.country_name(code) or loc.country

        def setp(lat: float, lon: float, precision: str) -> Location:
            loc.latitude, loc.longitude, loc.precision = lat, lon, precision
            return loc

        if self.nominatim and loc.place:
            hit = self.nominatim.lookup(", ".join(p for p in [loc.place, loc.city, loc.region, loc.country] if p))
            if hit:
                return setp(*hit, "place")

        hit3 = self.gaz.city(loc.city, loc.country)
        if hit3:
            if not loc.country:
                loc.country = self.gaz.country_name(hit3[2])
            return setp(hit3[0], hit3[1], "city")

        if self.nominatim and (loc.city or loc.region):
            q = ", ".join(p for p in [loc.city, loc.region, loc.country] if p)
            hit = self.nominatim.lookup(q)
            if hit:
                return setp(*hit, "city" if loc.city else "region")

        if llm_lat is not None:
            return setp(llm_lat, llm_lon, "llm-estimate")

        pt = self.gaz.country_point(loc.country)
        if pt:
            return setp(pt[0], pt[1], "country")

        loc.latitude = loc.longitude = None
        loc.precision = None
        return loc
