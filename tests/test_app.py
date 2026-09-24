import re
import time

import pytest

from app import db
from app.classifier import CATEGORY_NAMES, AdClassifier, build_text, keyword_scores
from app.scraper import _parse_google, _parse_taboola, ad_id, is_junk

REMOTE = {"REMOTE_ADDR": "203.0.113.7"}  # a customer on another computer
FORBIDDEN = ["scrap", "playwright", "taboola", "google ads", "tf-idf", "logistic", "chromium", "headless",
             "/admin", "/api/admin", "needs review", "ml+rules", "semantic", "retrain", "source_page", "network",
             "srchly", "needs_review", "manual_label"]


@pytest.fixture(scope="session")
def model():
    return AdClassifier.load()


@pytest.fixture(scope="session")
def seeded_db(tmp_path_factory):
    """Build the sample ad library once (categorising ~400 ads takes ~30 s), then copy it per test."""
    from app.web import create_app
    path = tmp_path_factory.mktemp("seed") / "seed.db"
    create_app(db_path=path, auto_refresh=False)
    return path


@pytest.fixture
def client(tmp_path, seeded_db):
    import shutil
    from app.web import create_app
    shutil.copy(seeded_db, tmp_path / "test.db")
    app = create_app(db_path=tmp_path / "test.db", auto_refresh=False)
    app.config["DB_PATH"] = tmp_path / "test.db"
    return app.test_client()


def _raw_ad(client):
    return db.query_ads(path=client.application.config["DB_PATH"])[0]


def _assert_clean(text: str, where: str):
    text = text.lower()
    # no ad-network servers anywhere (e.g. hot-linked images or click-tracking links)...
    for host in ("taboola.com", "doubleclick.net", "googlesyndication", "srchly.co"):
        assert not re.search(r"https?://[^\s\"'<>]*" + re.escape(host), text), (where, host)
    # ...then ignore advertisers' own landing-page URLs, which are the advertiser's content
    text = re.sub(r"https?://[^\s\"'<>,]*", "", text)
    for word in FORBIDDEN:
        assert word not in text, (where, word)


# ---------------------------------------------------------------- scraper
def test_taboola_sponsored_card_parsed():
    card = {"sponsored": True, "title": "Term Insurance for a Worry-Free Future", "description": "",
            "advertiser": "Policybazaar", "raw_text": "", "landing_url": "https://termlife.policybazaar.com/x",
            "image_url": ""}
    ad = _parse_taboola(card, "https://www.lokmattimes.com/", "2026-01-01T00:00:00+00:00")
    assert ad["domain"] == "termlife.policybazaar.com" and ad["network"] == "Taboola"


def test_taboola_organic_and_internal_links_skipped():
    base = {"title": "Some news", "description": "", "advertiser": "", "raw_text": "", "image_url": ""}
    assert _parse_taboola({**base, "sponsored": False, "landing_url": "https://x.com"}, "p", "t") is None
    assert _parse_taboola({**base, "sponsored": True, "landing_url": "https://www.lokmattimes.com/a"}, "p", "t") is None


def test_advertiser_fallback_from_raw_text():
    card = {"sponsored": True, "title": "", "description": "", "advertiser": "",
            "raw_text": "Go from beginner to confident trader\niFOREX|Sponsored\nSign Up",
            "landing_url": "https://www.iforex.in/lp", "image_url": ""}
    ad = _parse_taboola(card, "p", "t")
    assert ad["title"] == "Go from beginner to confident trader" and ad["advertiser"] == "iFOREX"


def test_google_click_url_resolved():
    data = {"text": "Ad\nBig Diwali Sale\nUp to 70% off", "alts": [],
            "links": ["https://adclick.g.doubleclick.net/pcs/click?x=1&adurl=https://www.flipkart.com/sale"]}
    ad = _parse_google(data, "p", "t")
    assert ad["domain"] == "flipkart.com" and ad["title"] == "Big Diwali Sale"


def test_junk_and_ids():
    assert is_junk("Click Here", "https://a.com") and is_junk("youtube.com", "https://youtube.com")
    assert not is_junk("Play War Thunder now for free", "https://warthunder.com")
    assert ad_id("A  b", "X", "https://www.x.com/1") == ad_id("a b", "x", "https://x.com/2")


def test_strip_tracking():
    from app.scraper import strip_tracking
    assert strip_tracking("https://x.com/p?utm_source=taboola&tblci=abc&id=7#top") == "https://x.com/p?id=7"


# ---------------------------------------------------------------- classifier
def test_build_text_splits_domain():
    assert "elitematrimony" in build_text("t", domain="www.elitematrimony.com")


def test_keyword_rules_devanagari():
    _, hits = keyword_scores("नवी मुंबईत लवकर उपलब्ध घरे")
    assert "Real Estate" in hits


@pytest.mark.parametrize("title,advertiser,expected", [
    ("Term insurance for your family, ₹1 crore cover", "Max Life", "Insurance"),
    ("Learn intraday trading strategy from experts", "TradeWise", "Finance & Investment"),
    ("Exclusive matchmaking for elite families", "Elite Matrimony", "Matrimony & Dating"),
    ("Hiring delivery executives in Pune, apply now", "", "Jobs & Careers"),
    ("3 BHK apartments in Wakad, new launch", "", "Real Estate"),
    ("This 87-Year-Old Woman Lives in a Tiny House - Take a Look Inside", "Travel Sent", "Sponsored Stories"),
    ("नई SUV पर भारी छूट, आज ही टेस्ट ड्राइव बुक करें", "", "Automotive"),
    ("पुण्यात १ व २ बीएचके फ्लॅट्स, नोंदणी सुरू", "", "Real Estate"),
    ("Kubernetes set up simply and securely", "OVHcloud", "Technology & Electronics"),
])
def test_predictions(model, title, advertiser, expected):
    assert model.predict(title, advertiser=advertiser)["category"] == expected


def test_prediction_shape(model):
    r = model.predict("Play the best free MMO shooter")
    assert r["category"] in CATEGORY_NAMES and 0 <= r["confidence"] <= 1 and len(r["alternatives"]) == 3


def test_advertiser_key_networks():
    from app.classifier import advertiser_key
    assert advertiser_key("Capsule Coffee", "srchly.co") != advertiser_key("Donate to Orphanage", "srchly.co")
    assert advertiser_key("Hearing Aids", "in.hear.com") == advertiser_key("Hearing Loss", "in.hear.com")


# ---------------------------------------------------------------- customer site
def test_pages_render(client):
    for url in ["/", "/ads", "/ads?page=2", "/ads?page=999", "/classify", "/live", "/live?section=/pune/",
                "/admin", "/ads?category=Gaming&q=war"]:
        assert client.get(url).status_code == 200, url


def test_customer_responses_reveal_no_internals(client, monkeypatch):
    """Everything a customer's browser receives: pages, JavaScript, CSS, JSON, CSV and error pages."""
    from app.live import Snapshotter
    monkeypatch.setattr(Snapshotter, "request", lambda self, section: "working")
    for url in ["/", "/ads", "/ads?page=2", "/classify", "/live", "/static/app.js", "/static/style.css",
                "/api/ads", "/api/stats", "/export.csv", "/nope"]:
        _assert_clean(client.get(url, environ_base=REMOTE).get_data(as_text=True), url)
    _assert_clean(client.post("/classify", data={"title": "3 BHK flats in Wakad"}).get_data(as_text=True), "classify")
    _assert_clean(client.post("/api/live", json={"section": "/"}).get_data(as_text=True), "api/live")


def test_public_ad_fields(client):
    ad = client.get("/api/ads").get_json()[0]
    assert set(ad) == {"id", "title", "description", "advertiser", "domain", "landing_url", "category",
                       "first_seen", "last_seen"}
    assert set(client.get("/api/stats").get_json()) == {"total_ads", "advertisers", "last_updated", "by_category"}


def test_ad_library_paginates(client):
    html = client.get("/ads").get_data(as_text=True)
    assert html.count('<article class="ad-card">') == 24 and "Page 1 of" in html


def test_friendly_404(client):
    r = client.get("/does-not-exist")
    assert r.status_code == 404 and "Page not found" in r.get_data(as_text=True)


def test_sample_data_loaded_and_export(client):
    assert client.get("/api/stats").get_json()["total_ads"] > 100
    r = client.get("/export.csv")
    assert r.status_code == 200 and b"Category,Headline" in r.data


def test_api_classify(client):
    r = client.post("/api/classify", json={"title": "Online MBA from a top university", "url": "https://upgrad.com"})
    assert set(r.get_json()) == {"category", "confidence", "alternatives"}
    assert r.get_json()["category"] == "Education & Courses"
    assert client.post("/api/classify", json={}).status_code == 400


# ---------------------------------------------------------------- staff protection
def test_staff_pages_hidden_from_remote_visitors(client):
    for url in ["/admin", "/api/admin/refresh/status", "/static/admin.js"]:
        if url.startswith("/static"):
            continue  # static file itself is harmless; the endpoints it calls are protected
        assert client.get(url, environ_base=REMOTE).status_code == 404
        assert client.get(url, headers={"X-Forwarded-For": "203.0.113.7"}).status_code == 404  # behind a proxy
    assert client.post("/api/admin/retrain", environ_base=REMOTE).status_code == 404
    assert client.post("/api/admin/refresh", environ_base=REMOTE).status_code == 404
    assert client.post("/api/admin/ads/x/label", json={"category": "Gaming"}, environ_base=REMOTE).status_code == 404


def test_staff_login_with_password(client):
    import base64
    client.application.config["ADMIN_PASSWORD"] = "s3cret-pass"
    assert client.get("/admin", environ_base=REMOTE).status_code == 401
    bad = {"Authorization": "Basic " + base64.b64encode(b"staff:wrong").decode()}
    good = {"Authorization": "Basic " + base64.b64encode(b"staff:s3cret-pass").decode()}
    assert client.get("/admin", environ_base=REMOTE, headers=bad).status_code == 401
    assert client.get("/admin", environ_base=REMOTE, headers=good).status_code == 200


def test_label_feedback_roundtrip(client, tmp_path):
    ad = _raw_ad(client)
    assert client.post(f"/api/admin/ads/{ad['id']}/label", json={"category": "Automotive"}).get_json()["ok"]
    assert client.post(f"/api/admin/ads/{ad['id']}/label", json={"category": "Nope"}).status_code == 400
    assert client.post("/api/admin/ads/missing/label", json={"category": "Automotive"}).status_code == 404
    # advertiser memory applies the correction to that advertiser's next ad straight away
    again = client.post("/api/classify", json={"title": ad["title"] + " today", "advertiser": ad["advertiser"],
                                               "domain": ad["domain"]}).get_json()
    assert again["category"] == "Automotive"
    fb = db.feedback_rows(tmp_path / "test.db")
    assert fb == [{"title": ad["title"], "description": ad["description"], "advertiser": ad["advertiser"],
                   "domain": ad["domain"], "category": "Automotive"}]


# ---------------------------------------------------------------- live site
def test_live_snapshot_api(client, monkeypatch):
    from app.live import Snapshotter
    asked = []
    monkeypatch.setattr(Snapshotter, "request", lambda self, section: asked.append(section) or "working")
    assert client.post("/api/live", json={"section": "/evil/"}).status_code == 400
    assert client.post("/api/live", json={"section": "/pune/"}).get_json() == {"state": "working"}
    assert asked == ["/pune/"]
    assert client.get("/api/live/status?section=/pune/").get_json() == {"state": "idle"}
    assert client.get("/live/img/..%2Fads.db.jpg").status_code == 404


def test_live_page_shows_snapshot(client):
    snaps = client.application.config["SNAPSHOTS"]
    snaps.results["home"] = {"section": "/", "key": "home", "taken_ts": time.time(), "taken_at": "24 Sep 2026, 03:00 PM",
                             "width": 1280, "height": 3000, "placements": 2, "image": "home.jpg", "version": 1,
                             "ads": [{"id": "a" * 16, "title": "3 BHK flats", "advertiser": "X", "category": "Real Estate",
                                      "color": "#bcbd22", "spots": [{"x": 1, "y": 2, "w": 3, "h": 4}] * 2}]}
    html = client.get("/live").get_data(as_text=True)
    assert "1 ad on this page" in html and "Real Estate" in html and "×2" in html


def test_live_window_is_staff_only(client, monkeypatch):
    from app.live import LiveView
    opened = []
    monkeypatch.setattr(LiveView, "open", lambda self, path="/": opened.append(path) or True)
    assert client.post("/api/admin/live-window", json={"section": "/pune/"}, environ_base=REMOTE).status_code == 404
    assert client.post("/api/admin/live-window", json={"section": "/pune/"}).get_json()["ok"]
    assert opened == ["/pune/"]


def test_live_labels_match_library(client):
    """Ads already in the library (incl. staff corrections) get the same label on the live site."""
    live = client.application.config["LIVE"]
    ad = _raw_ad(client)
    client.post(f"/api/admin/ads/{ad['id']}/label", json={"category": "Gaming"})
    new_ad = {"id": "0" * 16, "title": "3 BHK flats in Wakad, new launch", "description": "",
              "advertiser": "", "domain": ""}
    res = live.categorize([ad, new_ad])
    assert res[0]["category"] == "Gaming" and res[1]["category"] == "Real Estate"


def test_live_scripts_are_valid_js_shape():
    from app.live import LABEL_JS, POSITIONS_JS, SCAN_JS
    assert SCAN_JS.strip().startswith("() =>") and LABEL_JS.strip().startswith("([el, info]) =>")
    assert POSITIONS_JS.strip().startswith("() =>")
    assert "attachShadow" in LABEL_JS  # panel isolated from the site's CSS


def test_encoder_falls_back_when_preferred_model_missing(monkeypatch):
    """If the preferred language model can't be loaded, the next one in the list is used."""
    import app.classifier as C
    monkeypatch.setattr(C, "EMBED_MODELS", [("no-such-org/no-such-model", "")] + C.EMBED_MODELS[-1:])
    monkeypatch.setattr(C._Encoder, "_model", None)
    monkeypatch.setattr(C._Encoder, "name", None)
    monkeypatch.setattr(C._Encoder, "_tried", set())
    assert C._Encoder.available()
    assert C._Encoder.name == C.EMBED_MODELS[-1][0]
    assert C._Encoder.encode(["hello"]).shape[0] == 1


def test_trained_model_records_its_encoder(model):
    assert model.meta.get("embed_model") == "google/embeddinggemma-300m"


def test_marathi_short_keywords_match_whole_words_only():
    assert "Matrimony & Dating" not in keyword_scores("कपड्यांवर ५०% सवलत")[1]   # "वर" inside a word
    assert "Matrimony & Dating" not in keyword_scores("पैकर्स एंड मूवर्स")[1]
    assert "Matrimony & Dating" in keyword_scores("मराठी वधू-वर सूचक मंडळ")[1]
    assert "Real Estate" in keyword_scores("फ्लॅटसाठी नोंदणी सुरू")[1]              # suffixes still match


def test_marathi_hindi_heldout_accuracy(model):
    """63 Marathi/Hindi ads never used for training (data/eval_marathi_hindi.csv). Measured 93.7% on 24 Sep 2026."""
    import csv
    rows = list(csv.DictReader(open(db.ROOT / "data" / "eval_marathi_hindi.csv", encoding="utf-8")))
    ok = sum(model.predict(r["title"])["category"] == r["category"] for r in rows)
    assert ok / len(rows) >= 0.90, f"{ok}/{len(rows)}"
