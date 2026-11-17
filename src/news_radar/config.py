"""Runtime configuration. Everything can be set from environment variables / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return list(default)
    return [x.strip() for x in raw.split(",") if x.strip()]


# Google News country editions used for a worldwide sweep (English editions).
WORLD_COUNTRIES: List[str] = [
    "US", "GB", "CA", "AU", "NZ", "IE",          # Americas / Europe / Oceania
    "IN", "PK", "BD", "SG", "MY", "PH",          # South & South-East Asia
    "NG", "KE", "ZA", "GH", "UG", "TZ", "ZW",    # Africa
    "AE", "IL", "LB",                            # Middle East
]

DEFAULT_COUNTRIES: List[str] = ["US", "GB", "IN", "AU", "NG", "ZA", "KE", "PH", "AE", "CA"]

# Search terms that steer Google News toward incident reporting. Each one is a separate query.
DEFAULT_QUERIES: List[str] = [
    "explosion OR blast",
    "shooting OR gunmen",
    "attack killed",
    "stabbing",
    "protest clashes OR riot",
    "earthquake OR flood OR cyclone",
    "wildfire OR landslide",
    "plane crash OR train derailment",
    "ship attacked OR vessel seized OR piracy",
    "kidnapped OR hostage",
    "cyberattack OR ransomware",
    "outbreak OR poisoning",
]

PERIODS = ["1h", "6h", "12h", "1d", "2d", "7d"]


@dataclass
class LLMConfig:
    backend: str = field(default_factory=lambda: os.getenv("RADAR_LLM_BACKEND", "groq"))  # groq | ollama
    groq_model: str = field(default_factory=lambda: os.getenv("RADAR_GROQ_MODEL", "llama-3.3-70b-versatile"))
    ollama_model: str = field(default_factory=lambda: os.getenv("RADAR_OLLAMA_MODEL", "qwen2.5:7b-instruct"))
    ollama_base_url: Optional[str] = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL"))
    temperature: float = 0.1
    workers: int = 4
    tpm_limit: int = field(default_factory=lambda: int(os.getenv("RADAR_TPM_LIMIT", "12000")))
    triage_batch_size: int = 15
    max_chars: int = 3000          # article text sent to the analysis prompt
    max_retries: int = 5

    @property
    def model_name(self) -> str:
        return self.groq_model if self.backend == "groq" else self.ollama_model


@dataclass
class GeoConfig:
    use_nominatim: bool = field(default_factory=lambda: os.getenv("RADAR_USE_NOMINATIM", "false").lower() == "true")
    # Nominatim's policy requires an identifying User-Agent; set RADAR_GEOCODER_UA to include contact info.
    user_agent: str = field(default_factory=lambda: os.getenv("RADAR_GEOCODER_UA", "osint-news-radar/2.0"))
    min_delay_seconds: float = 1.1


@dataclass
class ScanConfig:
    # --- Discovery: global sweep (no source list required) ---
    global_scan: bool = True
    countries: List[str] = field(default_factory=lambda: _env_list("RADAR_COUNTRIES", DEFAULT_COUNTRIES))
    queries: List[str] = field(default_factory=lambda: list(DEFAULT_QUERIES))
    include_top_news: bool = False       # also pull each edition's top stories
    focus: str = ""                      # optional keyword appended to every query, e.g. "Nigeria"
    period: str = "1d"
    results_per_query: int = 8

    # --- Discovery: optional custom sources (home/section page URLs) ---
    sources: List[str] = field(default_factory=list)
    links_per_source: int = 15
    respect_robots: bool = True

    # --- Extraction ---
    extract_workers: int = 8
    request_timeout: int = 12
    max_articles: int = 150              # hard cap on articles extracted & triaged per run

    # --- Behaviour ---
    skip_seen: bool = True               # skip URLs already processed in previous runs
    min_confidence: float = 0.5          # triage confidence needed to treat as incident
    db_path: str = field(default_factory=lambda: os.getenv("RADAR_DB_PATH", "data/radar.db"))

    llm: LLMConfig = field(default_factory=LLMConfig)
    geo: GeoConfig = field(default_factory=GeoConfig)
