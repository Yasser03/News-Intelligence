import pytest

from news_radar.config import GeoConfig
from news_radar.discovery import canonical_url, title_key
from news_radar.geo import Gazetteer, Geocoder
from news_radar.llm import TokenBudget, parse_json
from news_radar.mapview import build_map, popup_html, safe_url, to_geojson
from news_radar.models import Incident, Location
from news_radar.taxonomy import TAXONOMY, normalize_category, normalize_severity


# ---------------------------------------------------------------- taxonomy
@pytest.mark.parametrize("cat,sub,expected", [
    ("Terrorism", "Bombing / IED", ("Terrorism", "Bombing / IED")),
    ("terrorism", "bombing", ("Terrorism", "Bombing / IED")),
    ("Natural Disasters", "Quake", ("Natural Disaster", "Earthquake")),
    (None, "Ransomware", ("Cyber Incident", "Ransomware")),
    ("Transport Accident", "spaceship", ("Transport Accident", "Other")),
    ("nonsense", "gibberish", ("Other", "Other")),
    ("", "", ("Other", "Other")),
])
def test_normalize_category(cat, sub, expected):
    assert normalize_category(cat, sub) == expected


def test_every_normalized_pair_is_valid():
    for cat, subs in TAXONOMY.items():
        for sub in subs:
            c, s = normalize_category(cat.upper(), sub.lower())
            assert (c, s) == (cat, sub)


@pytest.mark.parametrize("raw,expected", [
    ("high", "High"), ("CRITICAL", "Critical"), ("moderate", "Medium"), ("", "Low"), (None, "Low"), ("hgh", "High"),
])
def test_normalize_severity(raw, expected):
    assert normalize_severity(raw) == expected


# ---------------------------------------------------------------- llm json
def test_parse_json_variants():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('<think>hmm {"x":2}</think>\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! Here you go: {"a": [1,2]} hope it helps') == {"a": [1, 2]}
    assert parse_json('[{"id": 1}]') == [{"id": 1}]
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_token_budget_allows_oversized_request():
    b = TokenBudget(100)
    b.acquire(10_000)  # clamps to the limit instead of blocking forever
    assert b._used == 100


# ---------------------------------------------------------------- discovery utils
def test_canonical_url_strips_tracking_and_fragment():
    assert canonical_url("HTTPS://News.Example/a/b/?utm_source=x&id=5#top") == "https://news.example/a/b?id=5"


def test_title_key_ignores_outlet_suffix():
    assert title_key("Big fire in Rome - Reuters") == title_key("Big fire in Rome | BBC News")


# ---------------------------------------------------------------- geo
def test_gazetteer_city_and_country():
    g = Gazetteer.get()
    lat, lon, cc = g.city("Lagos", "Nigeria")
    assert cc == "NG" and 5 < lat < 8 and 2 < lon < 5
    assert g.city("Paris", "France")[2] == "FR"
    assert g.country_code("UK") == "GB" and g.country_code("Türkiye") in (None, "TR")
    assert g.country_point("Kenya") is not None


def test_geocoder_fallback_chain():
    geo = Geocoder(GeoConfig(use_nominatim=False))
    loc = geo.geocode(Location(city="Sapporo"))
    assert loc.precision == "city" and loc.country == "Japan"
    loc = geo.geocode(Location(country="Kenya", city="Tiny Village Nobody Knows", latitude=0.5, longitude=37.0))
    assert loc.precision == "llm-estimate" and loc.latitude == 0.5
    loc = geo.geocode(Location(country="Kenya"))
    assert loc.precision == "country"
    loc = geo.geocode(Location(country="Atlantis"))
    assert loc.latitude is None and loc.precision is None


# ---------------------------------------------------------------- map / export
def _incident(**kw):
    base = dict(id="1", url="https://x.example/a", category="Terrorism", subcategory="Bombing / IED",
                severity="High", headline="Blast", summary="Summary",
                location=Location(city="Kabul", country="Afghanistan", latitude=34.5, longitude=69.2,
                                  precision="city"))
    base.update(kw)
    return Incident(**base)


def test_popup_escapes_html_and_blocks_js_urls():
    inc = _incident(headline="<script>alert(1)</script>", url='javascript:alert("x")')
    html = popup_html(inc)
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "javascript:" not in html
    assert safe_url('https://ok.example/?a="b"') == "https://ok.example/?a=&quot;b&quot;"


def test_build_map_and_geojson():
    incs = [_incident(), _incident(id="2", location=Location(country="Nowhere"))]
    html = build_map(incs).get_root().render()
    assert "Terrorism" in html
    gj = to_geojson(incs)
    assert len(gj["features"]) == 1
    assert gj["features"][0]["geometry"]["coordinates"] == [69.2, 34.5]
