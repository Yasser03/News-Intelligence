# 📡 OSINT News Radar

**Worldwide news → LLM incident triage → categories & sub-categories → summaries on a live map.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scraping: pipescraper](https://img.shields.io/badge/scraping-pipescraper-orange.svg)](https://github.com/Yasser03/pipescraper)
[![LLM: Groq / Ollama](https://img.shields.io/badge/LLM-Groq%20%7C%20Ollama-green.svg)](https://console.groq.com/)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

News Radar scans news from around the world. You don't need a list of sources, though you can
add your own. An LLM decides which articles report a real security or safety **incident**, sorts
each incident into a **category and sub-category**, rates its severity, finds where it happened,
writes a short factual summary, and puts it on an interactive map.

There is no ML classifier and no training data. The LLM does all the classification.

```
              ┌───────────── pipescraper ─────────────┐
 Google News  │ FetchGoogleNews  (country × query)     │
 editions ────┤ FetchLinks       (optional sources)    ├─► dedupe ─► LLM triage ─► LLM analysis ─► geocode ─► SQLite ─► map / API / exports
 worldwide    │ ExtractArticles  (full text + time)    │   (URL +    Incident /    category,       gazetteer,
              └────────────────────────────────────────┘    title)    Non-Incident  sub-category,   Nominatim,
                                                                      (batched)     severity,       LLM estimate
                                                                                    location,
                                                                                    summary
```

---

## Features

- **Global by default.** pipescraper's `FetchGoogleNews` sweeps many Google News country editions
  with incident-focused search terms. You can narrow the radar with a *focus* keyword such as
  `Sahel` or `Red Sea`.
- **Optional custom sources.** Add home or section pages (as text or CSV). `FetchLinks` finds the
  article links on them, and `ExtractArticles` pulls the full text, publish date and time.
- **LLM step 1: triage.** Headlines go to the LLM in batches, and each one is labelled
  *Incident* or *Non-Incident* with a confidence score. Batching keeps cost and token use low.
- **LLM step 2: analysis.** For each incident, one JSON call returns the category, sub-category,
  severity, a neutral headline, a 3–5 sentence summary, the event date, the location (country,
  region, city, place), casualties and the actors involved.
- **Fixed taxonomy.** Every label is mapped back onto [the taxonomy](#incident-taxonomy), so the
  map never shows free-text categories.
- **Geocoding.** An offline GeoNames gazetteer is tried first. Nominatim can be added for street
  or landmark level, and is cached and rate-limited. If neither finds the place, the LLM's
  coordinate estimate is used, then the country capital. Every point records how precise it is.
- **Memory between scans.** SQLite stores incidents, the URLs already processed and the LLM
  responses. Re-running a scan only processes new articles and never pays twice for the same one.
- **Outputs.**
  - Streamlit dashboard: map, filters, incident feed and history.
  - CLI for cron jobs.
  - Exports to CSV, JSON and **GeoJSON**.
- **Robust.** One failed source, LLM call or geocode never stops a scan. Errors are collected
  and shown in the results.

---

## Quick start

```bash
pip install -e .
```

```bash
cp .env.example .env
```

Then set `GROQ_API_KEY` in `.env`, or set `RADAR_LLM_BACKEND=ollama` to use a local model.
You can get a Groq key from the [Groq Console](https://console.groq.com/keys).

### Dashboard

```bash
streamlit run streamlit_app.py
```

Press **Run scan**. The default settings scan 10 Google News editions over the last 24 hours with
12 incident queries.

### Command line

```bash
news-radar scan --period 6h --countries world --out incidents.geojson --map radar.html
```

Focus on one region, include your own sources too, and use a local LLM:

```bash
news-radar scan --focus "Red Sea" --sources data/sources.example.txt --backend ollama --model qwen2.5:7b-instruct
```

`python -m news_radar scan …` works the same way. Run `news-radar scan --help` to see every option.

### Docker

```bash
docker build -t news-radar .
```

```bash
docker run -p 8501:8501 --env-file .env -v "$PWD/data:/app/data" news-radar
```

---

## Incident taxonomy

| Category | Sub-categories |
|---|---|
| Armed Conflict | Armed Clash · Airstrike / Drone Strike · Shelling / Artillery · Military Operation · Ceasefire Violation |
| Terrorism | Bombing / IED · Suicide Attack · Armed Assault · Vehicle Ramming · Hostage Taking · Terror Plot / Arrest |
| Violent Crime | Shooting · Stabbing · Homicide · Assault · Sexual Violence · Armed Robbery · Kidnapping |
| Organized & Property Crime | Drug Trafficking · Human Trafficking · Smuggling · Gang Violence · Arson · Burglary / Theft · Fraud / Financial Crime |
| Civil Unrest | Protest · Riot · Strike / Labor Action · Political Violence · Coup / Unconstitutional Change |
| Transport Accident | Road Accident · Rail Accident · Aviation Incident · Maritime Incident · Piracy / Maritime Security |
| Natural Disaster | Earthquake · Flood · Storm / Cyclone · Wildfire · Landslide · Volcanic Activity · Tsunami · Extreme Heat / Drought |
| Industrial & Infrastructure | Explosion · Structure Fire · Building Collapse · Hazmat / Chemical Leak · Power Outage · Infrastructure Failure |
| Health Emergency | Disease Outbreak · Mass Poisoning · Food Contamination |
| Cyber Incident | Cyberattack · Ransomware · Data Breach · Service Disruption |

**Severity levels:**
- **Critical:** mass casualties or national-level impact
- **High:** fatalities or serious regional disruption
- **Medium:** injuries or localized disruption
- **Low:** minor, no casualties

To change the taxonomy, edit `src/news_radar/taxonomy.py`. The prompts, normalisation and map
colours all read from it.

---

## Using it as a library

```python
from news_radar import ScanConfig
from news_radar.pipeline import RadarPipeline
from news_radar.mapview import build_map

cfg = ScanConfig(period="12h", focus="Nigeria")
result = RadarPipeline(cfg).run()

for inc in result.incidents:
    print(inc.severity, inc.category, ">", inc.subcategory, "|", inc.location.label(), "|", inc.headline)

build_map(result.incidents).save("radar.html")
```

## Project layout

```
src/news_radar/
  config.py      settings (env / .env), default editions and incident queries
  discovery.py   pipescraper: FetchGoogleNews, FetchLinks, ExtractArticles + dedupe
  llm.py         Groq / Ollama client (JSON mode), token budget, retries, JSON repair
  analyzer.py    LLM triage (batched) and per-incident analysis prompts
  taxonomy.py    categories, sub-categories, severity, colours, normalisation
  geo.py         offline gazetteer, Nominatim (optional), fallbacks
  store.py       SQLite: incidents, seen URLs, LLM cache, geocode cache
  pipeline.py    end-to-end scan orchestration
  mapview.py     Folium map, popups, GeoJSON export
  cli.py         `news-radar scan`
  app.py         Streamlit dashboard
tests/           offline tests: fake LLM + fake pipescraper fetchers
```

## Development

```bash
pip install -e ".[dev]"
```

```bash
pytest -q
```

```bash
ruff check src tests
```

The tests run fully offline: the LLM and the pipescraper network calls are replaced by fakes.

## Notes & limits

- **Language:** pipescraper's Google News fetcher queries English editions. Custom sources can be
  in any language, and the LLM writes its output in English.
- **Google News:** pipescraper needs the `gnews` package for this, and it is installed with the
  project. Google sometimes changes its redirect format. URLs that can't be decoded are dropped
  and counted under `urls_undecoded` in the scan stats.
- **Nominatim:** this is the public OpenStreetMap geocoder. It is limited to 1 request per second
  and needs a User-Agent that identifies you (`RADAR_GEOCODER_UA`). For heavy use, host your own
  instance or use a paid geocoder.
- **Accuracy:** LLM output can be wrong, so check incidents before acting on them. Points marked
  `llm-estimate` or `country` are approximate, and are drawn with dashed outlines on the map.

See [`docs/PRODUCTION_ROADMAP.md`](docs/PRODUCTION_ROADMAP.md) for the path to a production service.

## License

MIT. Author: **Dr. Yasser Mustafa**.
