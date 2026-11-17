"""Incident taxonomy (categories -> sub-categories) and normalisation helpers.

The LLM is asked to pick from this list; ``normalize_category`` maps whatever it
returns back onto a valid (category, subcategory) pair so downstream code never
sees free-text labels.
"""

from __future__ import annotations

import difflib
import re
from typing import Dict, List, Optional, Tuple

OTHER = "Other"

TAXONOMY: Dict[str, List[str]] = {
    "Armed Conflict": [
        "Armed Clash",
        "Airstrike / Drone Strike",
        "Shelling / Artillery",
        "Military Operation",
        "Ceasefire Violation",
    ],
    "Terrorism": [
        "Bombing / IED",
        "Suicide Attack",
        "Armed Assault",
        "Vehicle Ramming",
        "Hostage Taking",
        "Terror Plot / Arrest",
    ],
    "Violent Crime": [
        "Shooting",
        "Stabbing",
        "Homicide",
        "Assault",
        "Sexual Violence",
        "Armed Robbery",
        "Kidnapping",
    ],
    "Organized & Property Crime": [
        "Drug Trafficking",
        "Human Trafficking",
        "Smuggling",
        "Gang Violence",
        "Arson",
        "Burglary / Theft",
        "Fraud / Financial Crime",
    ],
    "Civil Unrest": [
        "Protest",
        "Riot",
        "Strike / Labor Action",
        "Political Violence",
        "Coup / Unconstitutional Change",
    ],
    "Transport Accident": [
        "Road Accident",
        "Rail Accident",
        "Aviation Incident",
        "Maritime Incident",
        "Piracy / Maritime Security",
    ],
    "Natural Disaster": [
        "Earthquake",
        "Flood",
        "Storm / Cyclone",
        "Wildfire",
        "Landslide",
        "Volcanic Activity",
        "Tsunami",
        "Extreme Heat / Drought",
    ],
    "Industrial & Infrastructure": [
        "Explosion",
        "Structure Fire",
        "Building Collapse",
        "Hazmat / Chemical Leak",
        "Power Outage",
        "Infrastructure Failure",
    ],
    "Health Emergency": [
        "Disease Outbreak",
        "Mass Poisoning",
        "Food Contamination",
    ],
    "Cyber Incident": [
        "Cyberattack",
        "Ransomware",
        "Data Breach",
        "Service Disruption",
    ],
    OTHER: [OTHER],
}

SEVERITIES: List[str] = ["Critical", "High", "Medium", "Low"]

SEVERITY_GUIDE = (
    "Critical = mass casualties (10+ killed), national-level security impact, or major disaster; "
    "High = fatalities or serious regional disruption; "
    "Medium = injuries, significant damage or localized disruption, no deaths; "
    "Low = minor event, no casualties, minimal disruption."
)

# Map colours (folium named colours) per category.
CATEGORY_COLORS: Dict[str, str] = {
    "Armed Conflict": "darkred",
    "Terrorism": "red",
    "Violent Crime": "purple",
    "Organized & Property Crime": "darkpurple",
    "Civil Unrest": "orange",
    "Transport Accident": "blue",
    "Natural Disaster": "green",
    "Industrial & Infrastructure": "cadetblue",
    "Health Emergency": "pink",
    "Cyber Incident": "black",
    OTHER: "gray",
}

# Same palette as hex, for CircleMarkers / legends.
CATEGORY_HEX: Dict[str, str] = {
    "Armed Conflict": "#8b0000",
    "Terrorism": "#e41a1c",
    "Violent Crime": "#984ea3",
    "Organized & Property Crime": "#5e3c99",
    "Civil Unrest": "#ff7f00",
    "Transport Accident": "#377eb8",
    "Natural Disaster": "#4daf4a",
    "Industrial & Infrastructure": "#2c7fb8",
    "Health Emergency": "#f781bf",
    "Cyber Incident": "#333333",
    OTHER: "#999999",
}

SEVERITY_RADIUS: Dict[str, int] = {"Critical": 13, "High": 10, "Medium": 7, "Low": 5}


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


_CAT_BY_KEY = {_key(c): c for c in TAXONOMY}
_SUB_INDEX: Dict[str, Tuple[str, str]] = {}
for _cat, _subs in TAXONOMY.items():
    for _sub in _subs:
        _SUB_INDEX.setdefault(_key(_sub), (_cat, _sub))
        # Also index each part of "A / B" style names.
        for part in _sub.split("/"):
            _SUB_INDEX.setdefault(_key(part), (_cat, _sub))


def _closest(key: str, candidates: List[str], cutoff: float = 0.75) -> Optional[str]:
    """Fuzzy match, falling back to substring containment ("quake" -> "earthquake")."""
    match = difflib.get_close_matches(key, candidates, n=1, cutoff=cutoff)
    if match:
        return match[0]
    if len(key) >= 4:
        for c in candidates:
            if key in c or (len(c) >= 4 and c in key):
                return c
    return None


def normalize_category(category: Optional[str], subcategory: Optional[str]) -> Tuple[str, str]:
    """Return a valid ``(category, subcategory)`` pair from possibly messy LLM output."""
    cat_key, sub_key = _key(category or ""), _key(subcategory or "")

    cat = _CAT_BY_KEY.get(cat_key)
    if cat is None and cat_key:
        hit = _closest(cat_key, list(_CAT_BY_KEY))
        cat = _CAT_BY_KEY[hit] if hit else None

    if cat is not None and cat != OTHER:
        subs = TAXONOMY[cat]
        sub_keys = {_key(s): s for s in subs}
        if sub_key in sub_keys:
            return cat, sub_keys[sub_key]
        hit = _closest(sub_key, list(sub_keys)) if sub_key else None
        if hit:
            return cat, sub_keys[hit]
        # Sub-category belongs to another category? keep the category, take its first match there
        if sub_key in _SUB_INDEX and _SUB_INDEX[sub_key][0] == cat:
            return _SUB_INDEX[sub_key]
        return cat, OTHER

    # Category missing/unknown: infer from the sub-category alone.
    if sub_key in _SUB_INDEX:
        return _SUB_INDEX[sub_key]
    hit = _closest(sub_key, list(_SUB_INDEX)) if sub_key else None
    if hit:
        return _SUB_INDEX[hit]
    return OTHER, OTHER


def normalize_severity(value: Optional[str]) -> str:
    """Map free-text severity onto one of ``SEVERITIES`` (defaults to ``Low``)."""
    v = _key(value or "")
    by_lower = {s.lower(): s for s in SEVERITIES}
    if v in by_lower:
        return by_lower[v]
    aliases = {"severe": "High", "serious": "High", "moderate": "Medium", "minor": "Low", "extreme": "Critical"}
    if v in aliases:
        return aliases[v]
    hit = _closest(v, list(by_lower), cutoff=0.6) if v else None
    return by_lower[hit] if hit else "Low"


def taxonomy_prompt_block() -> str:
    """Compact taxonomy listing for inclusion in LLM prompts."""
    lines = []
    for cat, subs in TAXONOMY.items():
        lines.append(f"- {cat}: {', '.join(subs)}")
    return "\n".join(lines)
