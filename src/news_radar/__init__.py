"""OSINT News Radar: global news scanning, LLM incident triage, categorisation and mapping."""

__version__ = "2.0.0"

from news_radar.config import GeoConfig, LLMConfig, ScanConfig
from news_radar.models import Article, Incident, Location
from news_radar.taxonomy import SEVERITIES, TAXONOMY

__all__ = [
    "ScanConfig",
    "LLMConfig",
    "GeoConfig",
    "Article",
    "Incident",
    "Location",
    "TAXONOMY",
    "SEVERITIES",
    "__version__",
]
