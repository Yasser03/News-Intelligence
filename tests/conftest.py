import json
import os
import re
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from news_radar.config import ScanConfig  # noqa: E402

INCIDENT_WORDS = re.compile(r"explosion|shooting|earthquake|flood|stabb|riot|crash|ransomware", re.I)

ANALYSES = {
    "explosion": {
        "category": "Industrial & Infrastructure", "subcategory": "Explosion", "severity": "high",
        "headline": "Gas explosion destroys apartment block in Lagos", "summary": "A gas explosion hit a block.",
        "event_date": "2026-11-16",
        "location": {"country": "Nigeria", "region": "Lagos State", "city": "Lagos", "place": None,
                     "latitude": 6.5, "longitude": 3.4},
        "killed": 3, "injured": "12", "actors": ["Lagos State Emergency Management Agency"], "confidence": 0.9,
    },
    "earthquake": {
        "category": "natural disasters", "subcategory": "quake", "severity": "Critical",
        "headline": "Strong earthquake strikes Hokkaido", "summary": "A 7.1 quake struck.",
        "location": {"country": "Japan", "city": "Sapporo"}, "killed": 14, "injured": None, "confidence": 0.95,
    },
    "shooting": {
        "category": "Violent Crime", "subcategory": "Shooting", "severity": "Medium",
        "headline": "Two wounded in shooting", "summary": "Two people were wounded.",
        "location": {"country": "Atlantis", "city": None, "latitude": None}, "confidence": 0.8,
    },
}


class FakeLLM:
    """Deterministic stand-in for Groq/Ollama that speaks the radar's prompt protocol."""

    model_name = "fake:test"

    def __init__(self, fail_on: str = ""):
        self.calls = []
        self.fail_on = fail_on

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        if "screening world news" in prompt:
            items = re.findall(r"^\[(\d+)\] (.*)$", prompt, re.M)
            results = [
                {"id": int(i), "label": "Incident" if INCIDENT_WORDS.search(t) else "Non-Incident",
                 "confidence": 0.9}
                for i, t in items
            ]
            return "<think>reasoning</think>```json\n" + json.dumps({"results": results}) + "\n```"
        title = re.search(r"^Title: (.*)$", prompt, re.M).group(1).lower()
        if self.fail_on and self.fail_on in title:
            raise RuntimeError("provider exploded")
        for k, v in ANALYSES.items():
            if k in title:
                return "Here is the JSON: " + json.dumps(v)
        return json.dumps({"category": "Other", "subcategory": "Other", "severity": "Low", "summary": "x"})


PAGES = {
    "https://news.example/lagos-gas-explosion": ("Gas explosion rocks Lagos apartment block - Example News",
                                                 "A powerful gas explosion ..."),
    "https://news.example/lagos-gas-explosion-copy": ("Gas explosion rocks Lagos apartment block | Other Outlet",
                                                      "Same story elsewhere"),
    "https://news.example/hokkaido-earthquake": ("Earthquake of magnitude 7.1 hits Hokkaido", "Sapporo shaken"),
    "https://news.example/market-rally": ("Stocks rally as tech earnings beat forecasts", "Markets up"),
    "https://news.example/shooting": ("Shooting leaves two wounded downtown", "Police said"),
    "https://news.example/broken": None,  # extraction fails
}


def fake_google(country, period, n, query):
    if country == "XX":
        raise ConnectionError("edition unavailable")
    urls = list(PAGES)
    if country == "GB":
        urls.append("https://news.google.com/rss/articles/CBMiundecodable")
        urls = [u + "?utm_source=gn" for u in urls]  # tracking params must dedupe away
    return urls[:n]


def fake_links(url, n, respect_robots):
    return ["https://custom.example/flood-warning-river-burst"]


def fake_extract(urls, workers, timeout):
    out = []
    for u in urls:
        base = u.split("?")[0]
        if base == "https://custom.example/flood-warning-river-burst":
            out.append(SimpleNamespace(url=base, title="Flood: river bursts banks in Sylhet", text="Flooding...",
                                       description="", source="Custom", author="", date_published="2026-11-17",
                                       time_published="08:00:00", language="en"))
            continue
        page = PAGES.get(base)
        if page:
            out.append(SimpleNamespace(url=base, title=page[0], text=page[1], description="", source="Example",
                                       author="", date_published="2026-11-17", time_published="", language="en"))
    return out


@pytest.fixture
def cfg(tmp_path):
    c = ScanConfig()
    c.countries = ["US", "GB", "XX"]
    c.queries = ["explosion"]
    c.results_per_query = 50
    c.sources = ["https://custom.example/"]
    c.db_path = str(tmp_path / "radar.db")
    c.llm.workers = 2
    c.llm.triage_batch_size = 3
    c.geo.use_nominatim = False
    return c
