"""Tests for the full-page ad scanner (local engine helpers, AI engine parsing, and the /api/scan-page route)."""
import base64
import io
import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app import page_scan as ps


def _png(w=600, h=900, color="white"):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- text composition
def test_headline_is_the_biggest_lettering_not_the_first_line():
    lines = [{"text": "Call 9876543210", "conf": 0.9, "h": 14, "x": 0, "y": 5, "w": 100},
             {"text": "Her Happiness!", "conf": 0.9, "h": 60, "x": 0, "y": 40, "w": 300},
             {"text": "Ayurvedic care for women", "conf": 0.8, "h": 18, "x": 0, "y": 120, "w": 250}]
    t = ps.compose_text(lines)
    assert t["headline"] == "Her Happiness!"
    assert "Call 9876543210" in t["description"] and "Ayurvedic" in t["description"]


def test_garbage_lines_are_ignored_and_lower_reading_quality():
    good = [{"text": "Personal loan at 10.5%", "conf": 0.95, "h": 20, "x": 0, "y": 0, "w": 200}]
    junk = good + [{"text": "|| ;; 1", "conf": 0.1, "h": 20, "x": 0, "y": 30, "w": 50}]
    assert ps.compose_text(junk)["full_text"] == "Personal loan at 10.5%"
    assert ps.reading_quality(good) > ps.reading_quality(junk)


def test_news_and_ad_wording():
    ad, news = ps.text_evidence("Book now! Call 9876543210 www.example.com ₹ 499")
    assert ad >= 3 and news == 0
    ad, news = ps.text_evidence("लोकमत न्यूज नेटवर्क\nपुणे : मंत्री म्हणाले")
    assert news >= 2
    assert ps.text_evidence("जाहीर सूचना: निविदा मागविण्यात येत आहेत")[0] >= 1


# ---------------------------------------------------------------- layout helpers
def test_split_only_on_printed_rules_not_white_space():
    img = Image.new("L", (400, 400), 255)
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 380, 180], fill=120)        # two grey ads with a white gap between them
    d.rectangle([20, 220, 380, 380], fill=120)
    gray = np.array(img)
    assert ps.split_block(gray, [0, 0, 400, 400]) == [[0, 0, 400, 400]]      # white gap: no cut
    d.line([0, 200, 400, 200], fill=0, width=3)                                 # printed dividing rule
    parts = ps.split_block(np.array(img), [0, 0, 400, 400])
    assert len(parts) == 2 and parts[0][3] <= parts[1][1] + 1


def test_white_gap_splits_only_when_both_sides_look_like_ads():
    img = Image.new("L", (400, 400), 255)
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 380, 180], fill=120)
    d.rectangle([20, 220, 380, 380], fill=120)
    gray = np.array(img)
    both_ads = np.ones((400 // 24 + 1, 400 // 24 + 1))
    rgb = np.array(Image.new("RGB", (400, 400), "white"))
    rgb[20:180, 20:380] = (200, 30, 30); rgb[220:380, 20:380] = (30, 60, 200)    # red ad above a blue ad
    assert len(ps.split_block(gray, [0, 0, 400, 400], both_ads, 24, rgb=rgb)) == 2
    same = rgb.copy(); same[220:380, 20:380] = (200, 30, 30)                      # same palette: one ad
    assert ps.split_block(gray, [0, 0, 400, 400], both_ads, 24, rgb=same) == [[0, 0, 400, 400]]
    one_side = both_ads.copy(); one_side[200 // 24:, :] = 0.1          # lower half looks like news
    assert ps.split_block(gray, [0, 0, 400, 400], one_side, 24, rgb=rgb) == [[0, 0, 400, 400]]


def test_fragments_inside_a_bigger_or_full_page_ad_are_dropped():
    page = 1000 * 1000
    big = {"bbox": {"x": 0, "y": 0, "w": 900, "h": 900}}
    frag = {"bbox": {"x": 100, "y": 100, "w": 200, "h": 100}}
    other = {"bbox": {"x": 910, "y": 0, "w": 90, "h": 300}}
    kept = ps._drop_nested([frag, big, other], page)
    assert big in kept and other in kept and frag not in kept


def test_unreadable_upload_gives_a_friendly_error():
    r = ps.scan_page(b"not an image")
    assert r["ads"] == [] and "not a readable image" in r["error"]


# ---------------------------------------------------------------- AI engine
class _FakeBlock:
    def __init__(self, text): self.type, self.text = "text", text


class _FakeResponse:
    def __init__(self, payload):
        self.content = [_FakeBlock(json.dumps(payload))]
        self.stop_reason, self.model = "end_turn", "claude-opus-5-5"


def test_ai_scan_converts_boxes_to_page_pixels(monkeypatch):
    import anthropic
    from app import ai_scan

    sent = {}

    class FakeClient:
        def __init__(self, *a, **k):
            self.beta = type("B", (), {"messages": type("M", (), {"create": staticmethod(self.create)})()})()

        def create(self, **kw):
            sent.update(kw)
            return _FakeResponse({"ads": [{"box": [500, 250, 1000, 500], "headline": "Columbus NXG",
                                           "description": "Sports shoes", "advertiser": "Columbus", "language": "en",
                                           "category": "Fashion & Shopping", "category_confidence": 0.93,
                                           "legibility": 0.9, "publisher_promotion": False}]})

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    res = ai_scan.scan_page_ai(_png(600, 900))
    assert res["engine"] == "ai" and len(res["ads"]) == 1
    a = res["ads"][0]
    assert a["bbox"] == {"x": 300, "y": 225, "w": 300, "h": 225}
    assert a["ai_category"] == "Fashion & Shopping" and a["ocr_quality_score"] == 0.9
    assert sent["model"] == "claude-opus-5-5" and sent["output_config"]["format"]["type"] == "json_schema"
    assert sent["fallbacks"] == "default"


def test_ai_scan_raises_on_refusal_so_the_app_can_fall_back(monkeypatch):
    import anthropic
    from app import ai_scan

    class R(_FakeResponse):
        def __init__(self): super().__init__({"ads": []}); self.stop_reason = "refusal"

    class FakeClient:
        def __init__(self, *a, **k):
            self.beta = type("B", (), {"messages": type("M", (), {"create": staticmethod(lambda **kw: R())})()})()

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    with pytest.raises(RuntimeError):
        ai_scan.scan_page_ai(_png())


def test_ai_can_be_switched_off(monkeypatch):
    from app import ai_scan
    monkeypatch.setenv("SORTIFY_AI", "off")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert ai_scan.ai_available() is False


# ---------------------------------------------------------------- /api/scan-page
@pytest.fixture
def scan_client(tmp_path):
    from app.web import create_app
    app = create_app(db_path=tmp_path / "scan.db", auto_refresh=False)
    return app.test_client()


def _fake_local(raw):
    return {"ads": [{"headline": "2 BHK flats in Hinjewadi, possession soon", "description": "Book site visit",
                     "advertiser": "", "bbox": {"x": 10, "y": 10, "w": 200, "h": 150},
                     "ocr_quality_score": 0.82, "confidence": 0.82}],
            "total_ads": 1, "page_width": 600, "page_height": 900, "engine": "local", "error": None}


def _post(client):
    data = "data:image/png;base64," + base64.b64encode(_png()).decode()
    return client.post("/api/scan-page", json={"image_data": data})


def test_scan_route_falls_back_to_local_when_ai_fails(scan_client, monkeypatch):
    from app import ai_scan, page_scan
    monkeypatch.setattr(ai_scan, "ai_available", lambda: True)
    monkeypatch.setattr(ai_scan, "scan_page_ai", lambda raw: (_ for _ in ()).throw(RuntimeError("no network")))
    monkeypatch.setattr(page_scan, "scan_page", _fake_local)
    r = _post(scan_client).get_json()
    assert r["engine"] == "local" and "unavailable" in r["engine_note"]
    ad = r["ads"][0]
    assert ad["category"] == "Real Estate"
    assert ad["ocr_quality"] == "Good" and 0 <= ad["category_confidence"] <= 1
    assert ad["thumb"].startswith("data:image/jpeg;base64,")


def test_scan_route_combines_ai_and_model(scan_client, monkeypatch):
    from app import ai_scan
    res = _fake_local(None)
    res["ads"][0].update(ai_category="Real Estate", ai_confidence=0.9)
    res["engine"] = "ai"
    monkeypatch.setattr(ai_scan, "ai_available", lambda: True)
    monkeypatch.setattr(ai_scan, "scan_page_ai", lambda raw: dict(res))
    ad = _post(scan_client).get_json()["ads"][0]
    assert ad["category"] == "Real Estate" and ad["category_confidence"] >= 0.9
    assert "agree" in ad["basis"]
