"""
AI page scanner (optional): Claude reads the page image, finds every advertisement,
and transcribes it (English / Marathi / Hindi), returning boxes + text + a category.

Switched on automatically when Anthropic credentials are available
(ANTHROPIC_API_KEY, or an `ant auth login` profile). Set SORTIFY_AI=off to force
the local engine. Page images are sent to Anthropic's API; nothing is stored there
beyond Anthropic's standard retention.

Returns the same shape as app.page_scan.scan_page, so the rest of the app does not
care which engine ran.
"""
from __future__ import annotations

import base64
import io
import json
import os

from PIL import Image

from .classifier import CATEGORY_NAMES
from .page_scan import load_image

MODEL = os.environ.get("SORTIFY_AI_MODEL", "claude-opus-5-5")
MAX_SIDE = 2400          # long side sent to the API; enough to read small print, keeps the upload small

SYSTEM = """You analyse scanned or photographed newspaper pages (Lokmat, Lokmat Times and similar Indian papers, \
in English, Marathi and Hindi) and extract the advertisements.

An advertisement is paid or promotional content: display ads, classified ads, public notices, tenders, auction or \
legal notices, obituary/greeting insertions, and the publisher's own promotions (contests, subscription offers). \
News reports, editorials, columns, photographs with captions, headlines, mastheads, section headers, page numbers \
and weather boxes are NOT advertisements - leave them out even if they mention brands or prices.

For each advertisement report:
- box: its outer boundary as [x1, y1, x2, y2] on a 0-1000 scale of the page width and height (0,0 = top-left).
  Two ads printed next to each other are two separate ads.
- headline: the most prominent text exactly as printed, in its original script (do not translate).
- description: other important text (offer, product, prices, dates, contact), as printed, at most ~300 characters.
- advertiser: the brand, company or organisation placing the ad ("" if not shown).
- language: "en", "mr", "hi" or "mixed".
- category: exactly one of the allowed categories.
- category_confidence: 0-1, how sure you are of the category.
- legibility: 0-1, how clearly the ad's text can be read in this image (1 = perfectly clear).
- publisher_promotion: true if it is the newspaper promoting itself.

If the page has no advertisements, return an empty list. Be exhaustive: small ads in corners and narrow strips count."""

SCHEMA = {
    "type": "object",
    "properties": {
        "ads": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "box": {"type": "array", "items": {"type": "number"}},
                    "headline": {"type": "string"},
                    "description": {"type": "string"},
                    "advertiser": {"type": "string"},
                    "language": {"type": "string", "enum": ["en", "mr", "hi", "mixed"]},
                    "category": {"type": "string", "enum": CATEGORY_NAMES},
                    "category_confidence": {"type": "number"},
                    "legibility": {"type": "number"},
                    "publisher_promotion": {"type": "boolean"},
                },
                "required": ["box", "headline", "description", "advertiser", "language", "category",
                             "category_confidence", "legibility", "publisher_promotion"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["ads"],
    "additionalProperties": False,
}


def ai_available() -> bool:
    """True when the AI engine can be used: not switched off, SDK installed, credentials present."""
    if os.environ.get("SORTIFY_AI", "").lower() in ("off", "0", "false", "no"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    # an `ant auth login` profile also works with the zero-arg client
    return os.path.isdir(os.path.expanduser(os.path.join("~", ".config", "anthropic")))


def _encode(img: Image.Image) -> tuple[str, float]:
    s = min(1.0, MAX_SIDE / max(img.size))
    if s < 1:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii"), s


def scan_page_ai(image_bytes: bytes) -> dict:
    """Scan with Claude. Raises on API/network errors so the caller can fall back to the local engine."""
    import anthropic

    img = load_image(image_bytes)
    data, _ = _encode(img)
    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",                      # if the model declines, Anthropic re-runs it on a fallback model
        system=SYSTEM,
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
            {"type": "text", "text": "List every advertisement on this newspaper page."},
        ]}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("The AI engine declined this image.")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("The AI engine's answer was cut off.")
    text = next((b.text for b in response.content if b.type == "text"), "")
    found = json.loads(text).get("ads", [])

    W, H = img.size
    ads = []
    for a in found:
        box = a.get("box") or []
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = (min(1000.0, max(0.0, float(v))) for v in box)
        if x2 <= x1 or y2 <= y1:
            continue
        px = [round(x1 * W / 1000), round(y1 * H / 1000), round(x2 * W / 1000), round(y2 * H / 1000)]
        leg = min(1.0, max(0.0, float(a.get("legibility", 0))))
        ads.append({
            "headline": a.get("headline", "").strip()[:160],
            "description": a.get("description", "").strip()[:600],
            "advertiser": a.get("advertiser", "").strip()[:120],
            "full_text": (a.get("headline", "") + "\n" + a.get("description", "")).strip(),
            "lines": [l for l in (a.get("headline", ""), a.get("description", "")) if l],
            "language": a.get("language", ""),
            "bbox": {"x": px[0], "y": px[1], "w": px[2] - px[0], "h": px[3] - px[1]},
            "ocr_quality_score": round(leg, 3), "confidence": round(leg, 3),
            "ai_category": a.get("category"), "ai_confidence": min(1.0, max(0.0, float(a.get("category_confidence", 0)))),
            "publisher_promotion": bool(a.get("publisher_promotion")),
            "ad_likelihood": 0.95,
        })
    ads.sort(key=lambda a: (a["bbox"]["y"] // 40, a["bbox"]["x"]))
    for i, a in enumerate(ads):
        a["ad_index"] = i
    return {"ads": ads, "total_ads": len(ads), "page_width": W, "page_height": H, "engine": "ai",
            "model": response.model, "error": None}
