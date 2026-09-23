import pytest

from app import db
from app.classifier import CATEGORY_NAMES, AdClassifier, build_text, keyword_scores
from app.scraper import _parse_google, _parse_taboola, ad_id, is_junk


@pytest.fixture(scope="session")
def model():
    return AdClassifier.load()


@pytest.fixture
def client(tmp_path):
    from app.web import create_app
    app = create_app(db_path=tmp_path / "test.db")
    return app.test_client()


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
    ("This 87-Year-Old Woman Lives in a Tiny House - Take a Look Inside", "Travel Sent", "Sponsored Content / Clickbait"),
])
def test_predictions(model, title, advertiser, expected):
    assert model.predict(title, advertiser=advertiser)["category"] == expected


def test_prediction_shape(model):
    r = model.predict("Play the best free MMO shooter")
    assert r["category"] in CATEGORY_NAMES and 0 <= r["confidence"] <= 1 and len(r["alternatives"]) == 3


# ---------------------------------------------------------------- web + db
def test_pages_render(client):
    for url in ["/", "/ads", "/classify", "/model", "/ads?review=1", "/ads?category=Gaming&q=war"]:
        assert client.get(url).status_code == 200, url


def test_sample_data_loaded_and_export(client):
    stats = client.get("/api/stats").get_json()
    assert stats["total_ads"] > 100
    r = client.get("/export.csv")
    assert r.status_code == 200 and b"category" in r.data


def test_api_classify(client):
    r = client.post("/api/classify", json={"title": "Online MBA from a top university", "url": "https://upgrad.com"})
    assert r.get_json()["category"] == "Education & Courses"
    assert client.post("/api/classify", json={}).status_code == 400


def test_label_feedback_roundtrip(client, tmp_path):
    ad = client.get("/api/ads").get_json()[0]
    assert client.post(f"/api/ads/{ad['id']}/label", json={"category": "Automotive"}).get_json()["ok"]
    assert client.post(f"/api/ads/{ad['id']}/label", json={"category": "Nope"}).status_code == 400
    assert client.post("/api/ads/missing/label", json={"category": "Automotive"}).status_code == 404
    fb = db.feedback_rows(tmp_path / "test.db")
    assert fb == [{"title": ad["title"], "description": ad["description"], "advertiser": ad["advertiser"],
                   "domain": ad["domain"], "category": "Automotive"}]
