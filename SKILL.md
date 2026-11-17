---
name: OSINT News Radar
description: Scan worldwide news with pipescraper, triage incidents and categorise/summarise them with an LLM, and map them.
---

# OSINT News Radar

## Setup
```bash
pip install -e .
```

```bash
cp .env.example .env
```

Then set `GROQ_API_KEY`, or set `RADAR_LLM_BACKEND=ollama`.

## Recipes

### Scan the world from the command line
```bash
news-radar scan --period 6h --countries world --out incidents.geojson --map radar.html
```

### Focus on one region, with your own sources too
```bash
news-radar scan --focus "Sahel" --sources data/sources.example.txt --period 1d
```

### Run a scan from Python
```python
from news_radar import ScanConfig
from news_radar.pipeline import RadarPipeline

cfg = ScanConfig(period="12h", countries=["GB", "IE"], focus="London")
cfg.llm.backend = "ollama"
result = RadarPipeline(cfg).run()
incidents = [i for i in result.incidents if i.category == "Civil Unrest"]
```

### Use only pipescraper (no LLM)
```python
from pipescraper import FetchGoogleNews, ExtractArticles, ToDataFrame
df = FetchGoogleNews(country="NG", period="1d", search="explosion", max_results=10) >> ExtractArticles() >> ToDataFrame()
```

### Load past incidents
```python
from news_radar.store import RadarStore
recent = RadarStore("data/radar.db").load_incidents(since_hours=24)
```

### Dashboard
```bash
streamlit run streamlit_app.py
```
