"""Command-line interface: ``news-radar scan`` (or ``python -m news_radar scan``)."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from typing import List, Optional

from news_radar.config import DEFAULT_QUERIES, PERIODS, WORLD_COUNTRIES, ScanConfig
from news_radar.mapview import build_map, to_geojson


def _split(v: Optional[str]) -> List[str]:
    return [x.strip() for x in (v or "").split(",") if x.strip()]


def _read_sources(path: str) -> List[str]:
    """Plain text (one URL per line) or CSV with a ``link``/``url`` column."""
    with open(path, encoding="utf-8") as f:
        if path.lower().endswith(".csv"):
            rows = list(csv.DictReader(f))
            for col in ("link", "url", "Link", "URL"):
                if rows and col in rows[0]:
                    return [r[col].strip() for r in rows if r.get(col, "").strip()]
            raise SystemExit(f"{path}: CSV needs a 'link' or 'url' column")
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="news-radar", description="OSINT News Radar")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="Run one radar scan")
    s.add_argument("--period", default="1d", choices=PERIODS)
    s.add_argument("--countries", help="Comma-separated Google News editions (default: config). 'world' = all")
    s.add_argument("--queries", help="Comma-separated search queries (default: built-in incident queries)")
    s.add_argument("--focus", default="", help="Keyword appended to every query, e.g. 'Nigeria'")
    s.add_argument("--top-news", action="store_true", help="Also include each edition's top stories")
    s.add_argument("--per-query", type=int, default=8, help="Results per (country, query)")
    s.add_argument("--no-global", action="store_true", help="Disable the global Google News sweep")
    s.add_argument("--sources", help="Optional file of source URLs (.txt one per line, or .csv with 'link')")
    s.add_argument("--max-articles", type=int, default=150)
    s.add_argument("--backend", choices=["groq", "ollama"])
    s.add_argument("--model", help="Model name for the chosen backend")
    s.add_argument("--workers", type=int, default=4, help="Parallel LLM calls")
    s.add_argument("--nominatim", action="store_true", help="Use Nominatim for street/landmark geocoding")
    s.add_argument("--include-seen", action="store_true", help="Re-process URLs seen in earlier runs")
    s.add_argument("--db", help="SQLite path (default: data/radar.db)")
    s.add_argument("--out", help="Write incidents to .json, .geojson or .csv")
    s.add_argument("--map", help="Write an interactive HTML map")
    s.add_argument("-v", "--verbose", action="store_true")
    return p


def config_from_args(a: argparse.Namespace) -> ScanConfig:
    cfg = ScanConfig()
    cfg.period = a.period
    if a.countries:
        cfg.countries = list(WORLD_COUNTRIES) if a.countries.lower() == "world" else [c.upper() for c in _split(a.countries)]
    cfg.queries = _split(a.queries) if a.queries else list(DEFAULT_QUERIES)
    cfg.focus = a.focus
    cfg.include_top_news = a.top_news
    cfg.results_per_query = a.per_query
    cfg.global_scan = not a.no_global
    cfg.sources = _read_sources(a.sources) if a.sources else []
    cfg.max_articles = a.max_articles
    cfg.skip_seen = not a.include_seen
    if a.db:
        cfg.db_path = a.db
    if a.backend:
        cfg.llm.backend = a.backend
    if a.model:
        if cfg.llm.backend == "groq":
            cfg.llm.groq_model = a.model
        else:
            cfg.llm.ollama_model = a.model
    cfg.llm.workers = a.workers
    cfg.geo.use_nominatim = a.nominatim
    if not cfg.global_scan and not cfg.sources:
        raise SystemExit("Nothing to scan: enable the global sweep or pass --sources")
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from news_radar.pipeline import RadarPipeline

    cfg = config_from_args(args)

    def progress(stage: str, msg: str, frac: float) -> None:
        print(f"[{frac * 100:5.1f}%] {stage:<9} {msg}", file=sys.stderr)

    result = RadarPipeline(cfg).run(progress=progress)

    for inc in result.incidents:
        loc = inc.location.label() or "?"
        print(f"{inc.severity:<8} {inc.category} > {inc.subcategory} | {loc} | {inc.headline}")
    print(json.dumps(result.stats), file=sys.stderr)
    if result.errors:
        print(f"{len(result.errors)} non-fatal errors (use -v for details)", file=sys.stderr)
        for e in result.errors[:20]:
            logging.info("error: %s", e)

    if args.out:
        out = args.out.lower()
        if out.endswith(".geojson"):
            payload = json.dumps(to_geojson(result.incidents), ensure_ascii=False, indent=2)
            open(args.out, "w", encoding="utf-8").write(payload)
        elif out.endswith(".csv"):
            rows = [i.flat() for i in result.incidents]
            with open(args.out, "w", newline="", encoding="utf-8") as f:
                if rows:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
        else:
            payload = json.dumps([i.model_dump() for i in result.incidents], ensure_ascii=False, indent=2)
            open(args.out, "w", encoding="utf-8").write(payload)
    if args.map:
        build_map(result.incidents).save(args.map)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
