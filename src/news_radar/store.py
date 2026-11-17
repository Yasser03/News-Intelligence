"""SQLite persistence: incidents, processed URLs, LLM response cache and geocode cache."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, List, Optional, Tuple

from news_radar.models import Incident, canonical_url

SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    scanned_at TEXT NOT NULL,
    category TEXT, subcategory TEXT, severity TEXT,
    country TEXT, latitude REAL, longitude REAL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_incidents_scanned ON incidents(scanned_at);
CREATE INDEX IF NOT EXISTS ix_incidents_category ON incidents(category);

CREATE TABLE IF NOT EXISTS seen_urls (
    url TEXT PRIMARY KEY,
    label TEXT NOT NULL,           -- incident | non-incident | failed
    seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS geo_cache (query TEXT PRIMARY KEY, lat REAL, lon REAL, created_at TEXT NOT NULL);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RadarStore:
    """Thread-safe wrapper around a single SQLite connection (``:memory:`` supported for tests)."""

    def __init__(self, path: str = "data/radar.db"):
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---------------------------------------------------------------- incidents
    def save_incidents(self, incidents: Iterable[Incident]) -> int:
        rows = [
            (
                i.id, i.url, i.scanned_at, i.category, i.subcategory, i.severity,
                i.location.country, i.location.latitude, i.location.longitude,
                i.model_dump_json(),
            )
            for i in incidents
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO incidents VALUES (?,?,?,?,?,?,?,?,?,?)", rows
            )
            self._conn.commit()
        return len(rows)

    def load_incidents(self, since_hours: Optional[float] = None, limit: int = 5000) -> List[Incident]:
        sql, args = "SELECT data FROM incidents", []
        if since_hours:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).isoformat(timespec="seconds")
            sql += " WHERE scanned_at >= ?"
            args.append(cutoff)
        sql += " ORDER BY scanned_at DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [Incident.model_validate_json(r[0]) for r in rows]

    def count_incidents(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]

    # ---------------------------------------------------------------- seen urls
    def is_seen(self, url: str) -> bool:
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM seen_urls WHERE url=?", (canonical_url(url),)).fetchone()
        return row is not None

    def mark_seen(self, items: Iterable[Tuple[str, str]]) -> None:
        now = _now()
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO seen_urls VALUES (?,?,?)", [(canonical_url(u), label, now) for u, label in items]
            )
            self._conn.commit()

    # ---------------------------------------------------------------- llm cache
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            row = self._conn.execute("SELECT value FROM llm_cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache VALUES (?,?,?)",
                (key, json.dumps(value, ensure_ascii=False, default=str), _now()),
            )
            self._conn.commit()

    # ---------------------------------------------------------------- geo cache
    def get_geo(self, query: str) -> Optional[Tuple[Optional[float], Optional[float]]]:
        with self._lock:
            row = self._conn.execute("SELECT lat, lon FROM geo_cache WHERE query=?", (query,)).fetchone()
        return (row[0], row[1]) if row else None

    def put_geo(self, query: str, lat: Optional[float], lon: Optional[float]) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO geo_cache VALUES (?,?,?,?)", (query, lat, lon, _now()))
            self._conn.commit()
