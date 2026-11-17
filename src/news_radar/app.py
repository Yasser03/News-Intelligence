"""Streamlit dashboard for the OSINT News Radar.

Run with:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import json
import os
from typing import List

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from news_radar.config import DEFAULT_QUERIES, PERIODS, WORLD_COUNTRIES, ScanConfig
from news_radar.mapview import build_map, to_geojson
from news_radar.models import Incident
from news_radar.taxonomy import CATEGORY_HEX, SEVERITIES, TAXONOMY

# ---------------------------------------------------------------------------
# Sidebar: scan configuration
# ---------------------------------------------------------------------------

def sidebar_config() -> ScanConfig:
    cfg = ScanConfig()
    sb = st.sidebar
    sb.title("📡 News Radar")

    sb.header("1 · Discovery")
    cfg.global_scan = sb.toggle("Global news sweep (no sources needed)", value=True,
                                help="Searches Google News editions worldwide via pipescraper.")
    if cfg.global_scan:
        cfg.period = sb.select_slider("Time window", options=PERIODS, value="1d")
        world = sb.checkbox("All available editions", value=False)
        cfg.countries = list(WORLD_COUNTRIES) if world else sb.multiselect(
            "Google News editions", options=WORLD_COUNTRIES, default=cfg.countries)
        cfg.focus = sb.text_input("Focus (optional)", placeholder="e.g. Sahel, Red Sea, Karachi",
                                  help="Added to every search to narrow the radar to a place or topic.")
        with sb.expander("Search queries"):
            q = st.text_area("One query per line", value="\n".join(DEFAULT_QUERIES), height=220)
            cfg.queries = [x.strip() for x in q.splitlines() if x.strip()]
            cfg.include_top_news = st.checkbox("Also include top stories", value=False)
            cfg.results_per_query = st.slider("Results per query & edition", 3, 30, 8)

    with sb.expander("Custom sources (optional)"):
        txt = st.text_area("Home/section page URLs, one per line", height=100,
                           placeholder="https://www.example-news.com/world")
        up = st.file_uploader("…or upload CSV (column 'link' or 'url')", type=["csv", "txt"])
        sources: List[str] = [x.strip() for x in txt.splitlines() if x.strip()]
        if up is not None:
            try:
                df = pd.read_csv(up) if up.name.endswith(".csv") else pd.DataFrame(
                    {"link": io.StringIO(up.getvalue().decode()).read().splitlines()})
                col = next((c for c in df.columns if c.lower() in {"link", "url"}), None)
                if col:
                    sources += [str(x).strip() for x in df[col].dropna() if str(x).startswith("http")]
                else:
                    st.error("CSV needs a 'link' or 'url' column")
            except Exception as e:
                st.error(f"Could not read file: {e}")
        cfg.sources = list(dict.fromkeys(sources))
        cfg.links_per_source = st.slider("Links per source", 5, 50, 15)

    sb.header("2 · LLM")
    backend = sb.radio("Backend", ["groq", "ollama"], horizontal=True,
                       index=0 if cfg.llm.backend == "groq" else 1,
                       format_func=lambda b: "Groq (cloud)" if b == "groq" else "Ollama (local)")
    cfg.llm.backend = backend
    if backend == "groq":
        cfg.llm.groq_model = sb.text_input("Groq model", cfg.llm.groq_model)
        if not os.getenv("GROQ_API_KEY"):
            sb.warning("GROQ_API_KEY is not set (.env or environment).")
        cfg.llm.tpm_limit = sb.number_input("Tokens / minute budget", 2000, 500000, cfg.llm.tpm_limit, 1000)
    else:
        cfg.llm.ollama_model = sb.text_input("Ollama model", cfg.llm.ollama_model)
    cfg.llm.workers = sb.slider("Parallel LLM calls", 1, 16, 4)
    cfg.min_confidence = sb.slider("Min. incident confidence", 0.0, 1.0, 0.5, 0.05)

    sb.header("3 · Limits & geo")
    cfg.max_articles = sb.slider("Max articles per scan", 10, 1000, 150, 10)
    cfg.extract_workers = sb.slider("Extraction workers", 1, 32, 8)
    cfg.geo.use_nominatim = sb.checkbox("Street-level geocoding (Nominatim, slower)", value=cfg.geo.use_nominatim)
    cfg.skip_seen = sb.checkbox("Skip articles seen in earlier scans", value=True)
    return cfg


# ---------------------------------------------------------------------------
# Results rendering
# ---------------------------------------------------------------------------

def embed_html(html: str, height: int) -> None:
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:  # Streamlit < 1.52
        components.html(html, height=height)


def incidents_frame(incidents: List[Incident]) -> pd.DataFrame:
    if not incidents:
        return pd.DataFrame()
    df = pd.DataFrame([i.flat() for i in incidents])
    cols = ["severity", "category", "subcategory", "headline", "country", "city", "place", "event_date",
            "killed", "injured", "summary", "source", "url", "geo_precision", "confidence", "published",
            "actors", "latitude", "longitude", "scanned_at", "id"]
    return df[[c for c in cols if c in df.columns]]


def apply_filters(incidents: List[Incident], key: str) -> List[Incident]:
    c1, c2, c3, c4 = st.columns([3, 2, 2, 3])
    present = [c for c in TAXONOMY if any(i.category == c for i in incidents)]
    cats = c1.multiselect("Category", present, default=present, key=f"{key}_cat")
    sevs = c2.multiselect("Severity", SEVERITIES, default=SEVERITIES, key=f"{key}_sev")
    countries = sorted({i.location.country for i in incidents if i.location.country})
    ctry = c3.multiselect("Country", countries, key=f"{key}_ctry")
    text = c4.text_input("Search text", key=f"{key}_q").strip().lower()
    out = [i for i in incidents if i.category in cats and i.severity in sevs]
    if ctry:
        out = [i for i in out if i.location.country in ctry]
    if text:
        out = [i for i in out if text in f"{i.headline} {i.summary} {i.location.label()} {i.subcategory}".lower()]
    return out


def render_results(incidents: List[Incident], key: str) -> None:
    if not incidents:
        st.info("No incidents to show yet.")
        return
    shown = apply_filters(incidents, key)

    k = st.columns(5)
    k[0].metric("Incidents", len(shown))
    k[1].metric("Critical / High", sum(i.severity in {"Critical", "High"} for i in shown))
    k[2].metric("Countries", len({i.location.country for i in shown if i.location.country}))
    k[3].metric("Reported killed", sum(i.killed or 0 for i in shown))
    k[4].metric("On map", sum(i.location.has_coords() for i in shown))

    map_col, side = st.columns([3, 1])
    with map_col:
        # All LLM/article text in the map is HTML-escaped and links are restricted to http(s).
        embed_html(build_map(shown).get_root().render(), height=620)
    with side:
        st.markdown("**By category**")
        counts = pd.Series([i.category for i in shown]).value_counts()
        for cat, n in counts.items():
            st.markdown(
                f"<span style='color:{CATEGORY_HEX.get(cat, '#999')}'>●</span> {cat} — **{n}**",
                unsafe_allow_html=True)
        st.markdown("**Top sub-categories**")
        st.dataframe(pd.Series([i.subcategory for i in shown]).value_counts().head(10).rename("count"),
                     width="stretch")

    st.subheader("Incident feed")
    for i in shown[:200]:
        loc = i.location.label() or "Unknown location"
        with st.expander(f"[{i.severity}] {i.subcategory} — {i.headline}  ·  {loc}"):
            st.write(i.summary or "_No summary_")
            meta = [f"**Category:** {i.category} › {i.subcategory}"]
            if i.killed or i.injured:
                meta.append(f"**Casualties:** {i.killed or 0} killed, {i.injured or 0} injured")
            if i.actors:
                meta.append(f"**Actors:** {', '.join(i.actors)}")
            meta.append(f"**Date:** {i.event_date or i.published or 'unknown'} · **Source:** [{i.source}]({i.url})")
            st.markdown("  \n".join(meta))

    df = incidents_frame(shown)
    with st.expander("Table view"):
        st.dataframe(df, width="stretch", hide_index=True)
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇ CSV", df.to_csv(index=False).encode(), "incidents.csv", "text/csv", key=f"{key}_csv")
    d2.download_button("⬇ JSON", json.dumps([i.model_dump() for i in shown], ensure_ascii=False, indent=2).encode(),
                       "incidents.json", "application/json", key=f"{key}_json")
    d3.download_button("⬇ GeoJSON", json.dumps(to_geojson(shown), ensure_ascii=False).encode(),
                       "incidents.geojson", "application/geo+json", key=f"{key}_geo")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="OSINT News Radar", page_icon="📡", layout="wide")
    cfg = sidebar_config()
    st.title("📡 OSINT News Radar")
    st.caption("Worldwide news → LLM incident triage → categories & sub-categories → summaries on a live map")

    run = st.sidebar.button("🚀 Run scan", type="primary", width="stretch")
    tab_scan, tab_history, tab_log = st.tabs(["Latest scan", "History", "Run details"])

    if run:
        if not cfg.global_scan and not cfg.sources:
            st.sidebar.error("Enable the global sweep or add at least one source.")
        else:
            from news_radar.pipeline import RadarPipeline

            bar = st.progress(0.0, text="Starting…")
            try:
                pipeline = RadarPipeline(cfg)
                result = pipeline.run(progress=lambda stage, msg, f: bar.progress(min(f, 1.0), text=msg))
                st.session_state["result"] = result
            except Exception as e:
                st.error(f"Scan failed: {e}")
            finally:
                bar.empty()

    result = st.session_state.get("result")
    with tab_scan:
        if result is None:
            st.info("Configure the scan in the sidebar and press **Run scan**. "
                    "No source list is required — the radar sweeps global news by default.")
        else:
            if not result.incidents and result.errors:
                st.warning("The scan finished without incidents and reported errors:\n\n- "
                           + "\n- ".join(dict.fromkeys(e.split(": ", 1)[-1] for e in result.errors[:50])))
            render_results(result.incidents, "scan")

    with tab_history:
        hours = st.select_slider("Show incidents scanned in the last", [6, 24, 48, 72, 168, 720],
                                 value=24, format_func=lambda h: f"{h} h" if h < 48 else f"{h // 24} days")
        try:
            from news_radar.store import RadarStore

            history = RadarStore(cfg.db_path).load_incidents(since_hours=hours)
        except Exception as e:
            history = []
            st.error(f"Could not read history database: {e}")
        render_results(history, "hist")

    with tab_log:
        if result is None:
            st.write("No scan run in this session yet.")
        else:
            st.json(result.stats)
            if result.errors:
                st.warning(f"{len(result.errors)} non-fatal errors")
                st.code("\n".join(result.errors[:200]))


if __name__ == "__main__":
    main()
