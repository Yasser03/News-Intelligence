from conftest import FakeLLM, fake_extract, fake_google, fake_links

from news_radar.analyzer import IncidentAnalyzer
from news_radar.discovery import NewsDiscovery
from news_radar.models import Article
from news_radar.pipeline import RadarPipeline
from news_radar.store import RadarStore


def make_pipeline(cfg, llm=None):
    store = RadarStore(cfg.db_path)
    disc = NewsDiscovery(cfg, google_fetch=fake_google, links_fetch=fake_links, extract=fake_extract)
    return RadarPipeline(cfg, store=store, llm=llm or FakeLLM(), discovery=disc), store


def test_discovery_dedupes_and_survives_failures(cfg):
    disc = NewsDiscovery(cfg, google_fetch=fake_google, links_fetch=fake_links, extract=fake_extract)
    urls = disc.discover_urls()
    bases = [u.split("?")[0] for u, _ in urls]
    assert len(bases) == len(set(bases))                      # tracking-param duplicates removed
    assert not any("news.google.com" in u for u in bases)     # undecoded redirects dropped
    assert not any("facebook.com" in u for u in bases)        # social links dropped
    assert "https://custom.example/flood-warning-river-burst" in bases  # optional source used
    assert disc.stats.failed_tasks == 1 and disc.stats.urls_undecoded == 1

    arts = disc.extract(urls)
    titles = [a.title for a in arts]
    assert sum("Gas explosion" in t for t in titles) == 1     # cross-outlet duplicate removed
    assert all(a.discovered_via for a in arts)
    assert "Live updates" not in titles                       # near-empty page dropped


def test_full_scan_end_to_end(cfg):
    pipe, store = make_pipeline(cfg)
    res = pipe.run(progress=lambda *a: None)

    by_sub = {i.subcategory: i for i in res.incidents}
    assert set(by_sub) == {"Explosion", "Earthquake", "Shooting", "Other"}  # flood -> Other via fake LLM
    assert all(i.subcategory != "Stocks" for i in res.incidents)

    exp = by_sub["Explosion"]
    assert exp.category == "Industrial & Infrastructure" and exp.severity == "High"
    assert exp.killed == 3 and exp.injured == 12
    assert exp.location.precision == "city" and exp.location.country == "Nigeria"

    quake = by_sub["Earthquake"]
    assert quake.category == "Natural Disaster" and quake.severity == "Critical"
    assert quake.location.has_coords()

    assert by_sub["Shooting"].location.latitude is None      # unknown country -> unmapped, still kept

    assert res.incidents[0].severity == "Critical"           # sorted by severity
    assert res.stats["incidents_detected"] == 4 and res.stats["mapped"] == 2
    assert store.count_incidents() == 4
    assert len(store.load_incidents(since_hours=1)) == 4


def test_second_scan_skips_seen_and_uses_cache(cfg):
    llm = FakeLLM()
    pipe, store = make_pipeline(cfg, llm)
    pipe.run(progress=lambda *a: None)
    first_calls = len(llm.calls)
    res2 = pipe.run(progress=lambda *a: None)
    assert res2.incidents == []
    assert res2.stats["urls_skipped_seen"] > 0
    assert len(llm.calls) == first_calls                     # nothing re-sent to the LLM


def test_analysis_failure_is_isolated(cfg):
    pipe, store = make_pipeline(cfg, FakeLLM(fail_on="earthquake"))
    res = pipe.run(progress=lambda *a: None)
    assert "Earthquake" not in {i.subcategory for i in res.incidents}
    assert any("analysis failed" in e for e in res.errors)
    assert not store.is_seen("https://news.example/hokkaido-earthquake")  # retried next run


def test_triage_batches_and_keeps_order(cfg):
    llm = FakeLLM()
    an = IncidentAnalyzer(llm, cfg.llm)
    arts = [Article(url=f"https://e/{i}", title=t) for i, t in
            enumerate(["Explosion at plant", "Cat show", "Riot in city", "Recipe", "Train crash", "Ransomware hit"])]
    out = an.triage(arts)
    assert [a.url for a, _ in out] == [a.url for a in arts]
    assert [t.is_incident for _, t in out] == [True, False, True, False, True, True]
    assert len(llm.calls) == 2  # batch size 3
