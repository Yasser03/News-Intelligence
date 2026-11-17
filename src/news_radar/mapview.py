"""Folium map + GeoJSON / table exports for incidents."""

from __future__ import annotations

from html import escape
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse

import folium
from folium.plugins import MarkerCluster

from news_radar.models import Incident
from news_radar.taxonomy import CATEGORY_HEX, OTHER, SEVERITY_RADIUS, TAXONOMY


def safe_url(url: str) -> str:
    """Only allow http(s) links in popups; everything else becomes '#'."""
    try:
        return escape(url, quote=True) if urlparse(url).scheme in {"http", "https"} else "#"
    except Exception:
        return "#"


def popup_html(i: Incident) -> str:
    loc = i.location
    casualties = []
    if i.killed:
        casualties.append(f"{i.killed} killed")
    if i.injured:
        casualties.append(f"{i.injured} injured")
    rows = [
        f"<b style='font-size:13px'>{escape(i.headline or i.article_title)}</b>",
        f"<span style='color:{CATEGORY_HEX.get(i.category, '#999')}'>&#9679;</span> "
        f"{escape(i.category)} &rsaquo; {escape(i.subcategory)} &middot; <b>{escape(i.severity)}</b>",
        f"<i>{escape(loc.label() or 'Unknown location')}</i>"
        + (f" <small>({escape(loc.precision)})</small>" if loc.precision else ""),
    ]
    if casualties:
        rows.append(escape(", ".join(casualties)))
    if i.event_date or i.published:
        rows.append(f"<small>{escape(i.event_date or i.published)}</small>")
    rows.append(f"<p style='margin:6px 0'>{escape(i.summary[:600])}</p>")
    rows.append(f"<a href='{safe_url(i.url)}' target='_blank' rel='noopener noreferrer'>"
                f"{escape(i.source or 'Source')} &rarr;</a>")
    return "<div style='font-family:sans-serif;font-size:12px;max-width:360px'>" + "<br>".join(rows) + "</div>"


def build_map(incidents: Iterable[Incident], cluster: bool = True, tiles: str = "OpenStreetMap") -> folium.Map:
    pts = [i for i in incidents if i.location.has_coords()]
    if pts:
        lat = sum(i.location.latitude for i in pts) / len(pts)
        lon = sum(i.location.longitude for i in pts) / len(pts)
        zoom = 2 if len(pts) > 1 else 6
    else:
        lat, lon, zoom = 20.0, 0.0, 2
    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles=tiles, world_copy_jump=True)

    groups: Dict[str, Any] = {}
    for cat in TAXONOMY:
        if any(i.category == cat for i in pts):
            fg = folium.FeatureGroup(name=f"{cat}", show=True)
            fg.add_to(m)
            groups[cat] = MarkerCluster(options={"disableClusteringAtZoom": 8}).add_to(fg) if cluster else fg

    for i in pts:
        color = CATEGORY_HEX.get(i.category, CATEGORY_HEX[OTHER])
        folium.CircleMarker(
            location=[i.location.latitude, i.location.longitude],
            radius=SEVERITY_RADIUS.get(i.severity, 6),
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            dash_array="4" if i.location.precision in {"country", "llm-estimate"} else None,
            tooltip=escape(f"[{i.severity}] {i.subcategory}: {i.headline or i.article_title}")[:160],
            popup=folium.Popup(popup_html(i), max_width=380),
        ).add_to(groups.get(i.category, m))

    if groups:
        folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(_legend_html({i.category for i in pts})))
    return m


def _legend_html(categories: set) -> str:
    items = "".join(
        f"<div><span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
        f"background:{CATEGORY_HEX[c]};margin-right:6px'></span>{escape(c)}</div>"
        for c in TAXONOMY if c in categories
    )
    if not items:
        return ""
    return (
        "<div style='position:fixed;bottom:24px;left:12px;z-index:9999;background:rgba(255,255,255,.92);"
        "padding:8px 10px;border-radius:6px;font:12px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3)'>"
        "<b>Categories</b>" + items +
        "<div style='margin-top:4px;color:#555'>Size = severity &middot; dashed = approximate location</div></div>"
    )


def to_geojson(incidents: Iterable[Incident]) -> Dict[str, Any]:
    features: List[Dict[str, Any]] = []
    for i in incidents:
        if not i.location.has_coords():
            continue
        props = i.flat()
        props.pop("latitude", None)
        props.pop("longitude", None)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [i.location.longitude, i.location.latitude]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}
