# News Intelligence — Production Roadmap

## 1. What this project is for

**News Intelligence is an OSINT incident-monitoring tool.** It watches a list of news
sites, finds the articles that report security or safety incidents (crime, attacks,
protests, maritime, air and traffic events), and turns each one into a structured,
geolocated record (type, impact level, place, summary) shown on a map.

Today it is a **single-user Streamlit prototype**: one person clicks "Run", the pipeline
runs synchronously inside the web request, and the results are gone when the page reloads.

### Current pipeline

```
sources.csv ──► RSS/Atom discovery ──► article scrape ──► ML filter ──► LLM analysis ──► geocode ──► table + map
 (local path)    (feedparser/bs4)      (trafilatura)     (LogReg .pkl)  (Groq/Ollama)   (Nominatim)  (Streamlit/Folium)
```

| Stage | Code | Notes |
|---|---|---|
| Feed discovery | `FeedDiscoverer` — `src/news_intelligence.py:440` | Tries `<link rel=alternate>` then common paths |
| Extraction | `ArticleExtractor` — `src/news_intelligence.py:515` | Thread pool, trafilatura JSON |
| ML filter | `NewsBinaryClassifier` — `src/news_binary_classifier.py` | Bag-of-words + Logistic Regression, pickled |
| LLM analysis | `IncidentAnalyzer` — `src/news_intelligence.py:588` | One prompt → JSON (type, impact, location, summary) |
| Geocoding + map | `IncidentVisualizer` — `src/news_intelligence.py:334` | GeoText + Nominatim, Folium circles |

### The production goal

> A continuously running service that ingests news on a schedule, stores
> deduplicated incident records with coordinates in a database, exposes them through an
> API and a dashboard, and alerts users when relevant incidents happen in the areas they
> care about.

---

## 2. Where it stands today

### What is already good
- The hybrid design is right: a cheap classifier first, then the LLM only on candidates.
- One LLM call per article returns every field, with an LRU cache and a tokens-per-minute limiter.
- Ollama and Groq backends both work, so it can run locally for free or in the cloud.

### Bugs to fix first (verified in code)

| # | Issue | Where | Impact |
|---|---|---|---|
| 1 | **Keyword pre-filter regex uses stems with `\b` on both sides** (`injur\b`, `explos\b`, `stabb\b`, `robber\b`, `attack\b`). "injured", "explosion", "stabbing", "robbery" and "attacks" do **not** match. | `src/news_intelligence.py:79` | With the pre-filter on, real incidents are thrown away |
| 2 | **Unescaped URL in the map popup `href`.** A feed link containing `"` or a `javascript:` URL is injected into the HTML. | `src/news_intelligence.py:405,416` | XSS in the dashboard |
| 3 | `LLMCache` and `TpmRateLimiter` are shared by worker threads with **no locks**. `deque.remove` can raise, and the check-then-append in the limiter lets bursts go over the budget. | `src/news_intelligence.py:121,297` | Random crashes, Groq 429s |
| 4 | Sources CSV defaults to a **hard-coded Windows path**, and no sample CSV is shipped. | `src/news_intelligence.py:815` | The app stops on first launch for everyone else |
| 5 | **Model weights are not in the repo** (`*.pkl` is gitignored) and are loaded with `pickle` (runs arbitrary code). The library's default path is `../../TALOS/...`, and paths are joined with `+`, so they need a trailing slash. | `src/news_binary_classifier.py:96,126` | Not reproducible; a security risk |
| 6 | The incident `time` keeps only `HH:MM:SS` (the date is dropped), and the fallback parses arbitrary title text. `datetime.utcnow()` is deprecated. | `src/news_intelligence.py:208,982` | Wrong or misleading timestamps |
| 7 | Geocoding is **sequential at 1.5 s per row with no cache**. `city_to_country` scans about 25k cities on every call. | `src/news_intelligence.py:340,286` | 50 incidents take more than 75 s; this load breaks Nominatim's usage policy |
| 8 | The LLM output is parsed with a hand-rolled JSON cleanup, and `incident_type` is never checked against the allowed list. | `src/news_intelligence.py:113,711` | Silent "Other" fallbacks, free-text categories |

### Structural gaps
- Business logic is tied to Streamlit (`@st.cache_data` on core classes, `st.*` calls in the pipeline).
- No persistence, no scheduling, no event deduplication (the same incident from five outlets shows up five times).
- No tests (`tests/` is empty), CI, type checking, linting, packaging, container or pinned dependencies.
- `print` is used instead of logging. There are no metrics, and failures are swallowed by bare `except: pass`.
- The ML model has no training code, no evaluation data, no version and no metrics.
- Legacy code paths (`stage in ["gate","impact","details"]`) are still present.

---

## 3. Target architecture

```
                ┌──────────────┐   schedule (every N min)
                │  Scheduler   │──────────────────────────┐
                └──────────────┘                          ▼
┌───────────┐   ┌──────────────┐   queue   ┌──────────────────────────┐
│ Sources   │──►│ Ingest worker│──────────►│ Processing workers       │
│ (DB table)│   │ feeds+scrape │           │ classify → LLM → geocode │
└───────────┘   └──────────────┘           │ → dedupe/cluster events  │
                                           └────────────┬─────────────┘
                                                        ▼
                     ┌───────────────────────────────────────────────┐
                     │ PostgreSQL + PostGIS (+ pgvector)             │
                     │ sources · articles · incidents · events · geo │
                     └───────┬──────────────────────┬────────────────┘
                             ▼                      ▼
                    ┌────────────────┐     ┌────────────────┐
                    │ FastAPI (REST) │     │ Alerting       │
                    │ auth, filters  │     │ email/Slack/   │
                    └───────┬────────┘     │ webhook by geo │
                            ▼              └────────────────┘
                    ┌────────────────┐
                    │ Dashboard      │  (Streamlit first, React + MapLibre later)
                    └────────────────┘
```

### Proposed repo layout

```
news_intelligence/            # installable package (pyproject.toml)
  config.py                   # pydantic-settings, all env-driven
  models.py                   # Pydantic: Article, IncidentAnalysis, Incident, Event
  ingest/feeds.py             # discovery + parsing (no Streamlit)
  ingest/extract.py           # trafilatura, robots.txt, per-domain rate limit
  classify/classifier.py      # ML filter (skops/ONNX artifact, versioned)
  classify/train.py           # reproducible training + evaluation
  analyze/llm.py              # structured output, provider abstraction, retry
  geo/geocode.py              # cached geocoder, gazetteer index
  dedupe/cluster.py           # embedding + time/geo clustering into events
  storage/                    # SQLAlchemy models, Alembic migrations, repositories
  pipeline.py                 # orchestrates one run; pure, testable
  workers/                    # task definitions (Celery/RQ/Arq) + scheduler
api/                          # FastAPI app
dashboard/                    # Streamlit app (UI only, reads from API/DB)
tests/unit, tests/integration, tests/fixtures (saved HTML/RSS, LLM responses)
deploy/                       # Dockerfile(s), docker-compose.yml, k8s/helm (later)
.github/workflows/ci.yml
```

### Technology choices

| Concern | Recommendation | Why |
|---|---|---|
| Config | `pydantic-settings` + `.env` | Typed settings, no hard-coded paths |
| Storage | PostgreSQL + PostGIS (+ pgvector) | Geo queries ("within 20 km of X"), embeddings for dedup |
| Jobs | Celery or RQ + Redis (Arq if going async) | Retries, schedules, horizontal scaling |
| HTTP | `httpx` with per-domain limits, ETag/If-Modified-Since | Polite and efficient crawling |
| LLM | LangChain `with_structured_output(PydanticModel)` or provider JSON mode | Removes the brittle JSON parsing |
| ML artifacts | `skops` or ONNX + model registry (MLflow) | Safe loading, versioning, metrics |
| Geocoding | Self-hosted Nominatim/Pelias, or a paid API, + DB cache | Policy-compliant, fast, deterministic |
| API | FastAPI + OAuth2/API keys | OpenAPI docs, async, easy to test |
| Observability | `structlog` JSON logs, Prometheus metrics, OpenTelemetry traces, Sentry | Know when feeds or the LLM break |
| Packaging | `pyproject.toml`, `uv`/pip-tools lock file, Docker multi-stage | Reproducible builds |

---

## 4. Phased plan

### Phase 0 — Stabilise the prototype (about 1 week)
- [ ] Fix bugs 1–8 from §2. For #1, use `\b(?:injur\w*|explos\w*|stabb\w*|robber\w*|attack\w*|…)` and add unit tests.
- [ ] Ship `data/sources.example.csv`; read the path from `NEWS_SOURCES_CSV`, falling back to an upload widget.
- [ ] Add `threading.Lock` to the cache and limiter (they move to Redis in Phase 2).
- [ ] Escape every value in the popup and allow only `http(s)` URLs.
- [ ] Keep the full ISO-8601 UTC timestamp. Prefer the publish date from the feed or metadata over parsing text.
- [ ] Replace `print` with `logging`, and bare `except` with logged exceptions plus error counters.
- [ ] Pin dependencies; add `ruff`, `mypy`, `pytest` and a GitHub Actions CI workflow.
- [ ] Delete the legacy stage branches and commented-out code.

### Phase 1 — Separate core from UI and add persistence (2–3 weeks)
- [ ] Move the logic into the `news_intelligence/` package with no Streamlit imports. The Streamlit app only calls `pipeline.run()`.
- [ ] Define Pydantic models for `Article`, `IncidentAnalysis` (enums for type and impact) and `Incident`.
- [ ] Switch the LLM to structured output. Validate it, retry once on failure, and record `model`, `prompt_version` and tokens per record.
- [ ] Add PostgreSQL/PostGIS with Alembic: `sources`, `articles` (URL hash plus content fingerprint unique), `incidents`, `geocode_cache`.
- [ ] Make it idempotent: skip already-processed URLs and fingerprints, so re-running is cheap.
- [ ] Add a CLI (`news-intel run --sources … --since 1h`) so it can run from cron before workers exist.
- [ ] Add a Dockerfile and a `docker-compose.yml` (app, postgres, redis, ollama).

### Phase 2 — Continuous, scalable processing (2–4 weeks)
- [ ] Add a scheduler and task queue: ingest per source → extract per URL → analyse per article, each as its own task with retries and dead-letter handling.
- [ ] Crawl politely: honour robots.txt, per-domain concurrency, conditional GETs and a real User-Agent with a contact address.
- [ ] Add a distributed rate limiter and LLM cache in Redis, plus a cost budget with alerts.
- [ ] Geocode with a cache-first lookup, an in-memory gazetteer index, and a self-hosted or commercial geocoder. Store a confidence and precision level (country, city or street).
- [ ] **Event deduplication:** embed each incident, then cluster by similarity + time window + distance, so one real event links to many articles.
- [ ] Add FastAPI endpoints: `GET /incidents?bbox=&since=&type=&impact=`, `GET /events/{id}`, `GET /sources/health`.
- [ ] Point the dashboard at the API: time slider, filters, clustering, heatmap, and source health.

### Phase 3 — ML quality and MLOps (ongoing, starting alongside Phase 2)
- [ ] Commit training code, a dataset card and an evaluation set. Report precision and recall for the `incident` class (recall matters most for a filter).
- [ ] Replace the 62k bag-of-words LogReg with TF-IDF + LogReg as a baseline, then a small multilingual transformer (e.g. distilled XLM-R) exported to ONNX.
- [ ] Use a confidence threshold instead of a hard label. Send borderline cases to the LLM.
- [ ] Support languages other than English: translate or use a multilingual model instead of dropping non-English articles (`langdetect` filter).
- [ ] Build an LLM evaluation harness: a golden set of articles with expected type, impact and location, run in CI whenever the prompt or model changes.
- [ ] Close the feedback loop: analysts correct records in the UI, and the corrections feed retraining.
- [ ] Use a model registry (MLflow), and record the model version on every prediction.

### Phase 4 — Operate and secure (before external users)
- [ ] Authentication and roles for the API and dashboard; per-tenant source lists and watch areas.
- [ ] Keep secrets in a secret manager, not `.env` in containers. Rotate the Groq key.
- [ ] Observability: dashboards for feeds OK/failed, articles/min, LLM latency, cost and parse-failure rate, geocode hit rate, and queue depth. Alert on SLO breaches.
- [ ] Alerting: users define areas (polygon or radius) plus type/impact rules, and get email, Slack or webhook notifications.
- [ ] Deploy with staging and prod environments, migrations in CI/CD, backups, and point-in-time recovery for Postgres.
- [ ] Load test at the target volume (e.g. 10k articles/hour) and size the workers.
- [ ] Legal and compliance: respect each site's terms of service, store excerpts rather than full text where required, follow geocoder usage policies, and set data retention.

---

## 5. Definition of "production ready"

| Area | Done when |
|---|---|
| Reliability | Runs unattended for 7 days; failed feeds and tasks retry and are visible; no data loss on restart |
| Correctness | Classifier recall ≥ agreed target on the held-out set; LLM field accuracy is tracked in CI; same incident reported once |
| Performance | Article reaches the dashboard in under 5 min (p95) at target volume |
| Security | No pickle loading of untrusted files, all output escaped, auth on all endpoints, secrets managed, dependency scanning in CI |
| Operability | One-command local stack (`docker compose up`), documented runbook, metrics and alerts |
| Quality | ≥ 80 % unit test coverage on core package; integration tests use recorded fixtures (no live network/LLM) |
| Cost | LLM spend per 1k articles measured, with a budget and alerts |

---

## 6. Suggested first five PRs

1. **Fix the pre-filter regex, popup escaping and thread safety**, with unit tests (bugs #1–#3).
2. **Config and sample data**: `pydantic-settings`, `data/sources.example.csv`, remove hard-coded paths, add logging.
3. **Tooling**: `pyproject.toml`, pinned lock file, ruff, mypy, pytest, GitHub Actions.
4. **Structured LLM output**: Pydantic schema with enums, replacing `safe_parse_json`.
5. **Extract the core package** out of the Streamlit file and add the `news-intel run` CLI.
