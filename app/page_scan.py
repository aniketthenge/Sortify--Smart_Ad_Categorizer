"""
Full-page ad scanner (local engine).

Finds the advertisements on a scanned or photographed newspaper page, reads the
text of each one, and leaves news out.

How it works
1. Normalise the upload: fix phone rotation (EXIF), work at a standard size
   (small or blurry uploads are enlarged and sharpened, huge ones shrunk).
2. Find candidate blocks from the *layout*, not the text:
   - boxes drawn with rules/borders (most print ads and notices are boxed),
   - colourful or picture-heavy areas (display ads),
   - dark filled panels (e.g. black-background ads).
3. Read each candidate separately: the crop is enlarged before OCR, which is
   far more accurate than reading the whole page at once.
4. Decide ad vs news for every block from layout + text evidence
   (news = many lines of small, even body text, datelines, "said"...;
    ads = large display lettering, pictures/colour, prices, phone numbers,
    web addresses, "call", "offer", public-notice wording...).
5. Pick the headline by lettering size (the biggest words), not just "first line".

The OCR engine (EasyOCR) is loaded once and reused.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter, ImageOps

WORK_LONG_SIDE = 2000        # layout analysis resolution
OCR_MIN_WIDTH = 1000         # crops are enlarged to at least this width before reading
OCR_MAX_WIDTH = 1800
MIN_BLOCK_AREA = 0.012       # ignore blocks smaller than 1.2% of the page
MAX_BLOCK_AREA = 0.97


@dataclass
class Block:
    x1: int
    y1: int
    x2: int
    y2: int
    source: str                      # "box", "graphic", "page"
    color_frac: float = 0.0
    lines: list = field(default_factory=list)   # (text, conf, height, rect)
    ad_score: float = 0.0
    reasons: list = field(default_factory=list)

    @property
    def area(self):
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def rect(self):
        return (self.x1, self.y1, self.x2, self.y2)


# ---------------------------------------------------------------------------- image helpers
def load_image(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)            # phone photos come rotated
    return img.convert("RGB")


def _to_work(img: Image.Image) -> tuple[Image.Image, float]:
    s = WORK_LONG_SIDE / max(img.size)
    if abs(s - 1) < 0.05:
        return img, 1.0
    return img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS), s


def _overlap(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


# ---------------------------------------------------------------------------- 1. candidate blocks
def find_candidate_blocks(work: Image.Image) -> list[Block]:
    import cv2

    arr = np.array(work)
    H, W = arr.shape[:2]
    page_area = W * H
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    # ink mask (dark on light)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 15)

    # ruled lines: long thin horizontal / vertical runs of ink
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(W // 22, 20), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(H // 22, 20)))
    lines = cv2.bitwise_or(cv2.morphologyEx(ink, cv2.MORPH_OPEN, hk), cv2.morphologyEx(ink, cv2.MORPH_OPEN, vk))

    # colour / picture mask: saturated pixels, plus large dark filled panels
    sat = ((hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 45)).astype(np.uint8) * 255
    dark = (gray < 60).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (W // 60, W // 60)))
    graphic = cv2.bitwise_or(sat, dark)
    color_integral = cv2.integral((graphic > 0).astype(np.uint8))

    def color_frac(x1, y1, x2, y2):
        tot = color_integral[y2, x2] - color_integral[y1, x2] - color_integral[y2, x1] + color_integral[y1, x1]
        return float(tot) / max(1, (x2 - x1) * (y2 - y1))

    blocks: list[Block] = []

    # (a) closed boxes formed by rules
    closed = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        a = w * h / page_area
        if not (MIN_BLOCK_AREA <= a <= MAX_BLOCK_AREA) or w < W * 0.08 or h < H * 0.04:
            continue
        # must really be a rectangle: rule pixels along most of its perimeter
        peri_hits = 0
        for (sx, sy, ex, ey) in ((x, y, x + w, y + 3), (x, y + h - 3, x + w, y + h), (x, y, x + 3, y + h), (x + w - 3, y, x + w, y + h)):
            peri_hits += (lines[sy:ey, sx:ex] > 0).mean() > 0.55
        if peri_hits >= 3:
            blocks.append(Block(x, y, x + w, y + h, "box"))

    # (b) colourful / picture areas grown into blocks
    grown = cv2.morphologyEx(graphic, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (W // 45, W // 45)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(grown)
    for i in range(1, n):
        x, y, w, h, area_px = stats[i]
        a = w * h / page_area
        if a < MIN_BLOCK_AREA or area_px < 0.35 * w * h:   # needs to be a solid-ish region
            continue
        blocks.append(Block(x, y, x + w, y + h, "graphic"))

    for b in blocks:
        b.color_frac = color_frac(b.x1, b.y1, b.x2, b.y2)

    # (c) whole page (for full-page ads)
    page = Block(0, 0, W, H, "page", color_frac=color_frac(0, 0, W, H))
    return _dedupe(blocks), page


def _dedupe(blocks: list[Block]) -> list[Block]:
    """Merge near-duplicates; when a graphic blob sits inside a ruled box, keep the box."""
    blocks = sorted(blocks, key=lambda b: -b.area)
    kept: list[Block] = []
    for b in blocks:
        dup = False
        for k in kept:
            ov = _overlap(b.rect(), k.rect())
            if ov >= 0.8 * b.area and ov >= 0.6 * k.area:      # essentially the same region
                dup = True
                break
        if not dup:
            kept.append(b)
    return kept


# ---------------------------------------------------------------------------- learned layout model
def regions_from_map(prob: np.ndarray, scale: float, cell: int, threshold: float = 0.5,
                     min_area: float = MIN_BLOCK_AREA) -> list[list[float]]:
    """Turn a per-cell 'ad likelihood' map into boxes [x1, y1, x2, y2] in original-image pixels."""
    import cv2

    rows, cols = prob.shape
    smooth = cv2.GaussianBlur(prob.astype(np.float32), (5, 5), 0)
    mask = (smooth >= threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w * h < min_area * rows * cols or area < 0.4 * w * h:
            continue
        boxes.append([x * cell / scale, y * cell / scale, (x + w) * cell / scale, (y + h) * cell / scale])
    return boxes


def _separator_runs(profile: np.ndarray, min_run: int) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, v in enumerate(profile):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_run:
                runs.append((start, i))
            start = None
    if start is not None and len(profile) - start >= min_run:
        runs.append((start, len(profile)))
    return runs


def _palette_distance(rgb: np.ndarray, a: list[int], b: list[int]) -> float:
    """0..1 difference between the colour make-up of two regions (hue/saturation/brightness histograms)."""
    import cv2
    def hist(r):
        crop = rgb[r[1]:r[3], r[0]:r[2]]
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256])
        return cv2.normalize(h, h).flatten()
    return float(cv2.compareHist(hist(a), hist(b), cv2.HISTCMP_BHATTACHARYYA))


def split_block(gray: np.ndarray, box: list[int], prob: np.ndarray | None = None, cell: int = 24,
                threshold: float = 0.5, depth: int = 0, rgb: np.ndarray | None = None) -> list[list[int]]:
    """Recursively cut a block where two ads were printed next to each other (work-image pixels).

    A cut must run all the way across the block, along either a printed rule, or - when a layout
    likelihood map is given - a clean white gap with an ad-looking part on BOTH sides
    (white space also occurs inside ads, so a gap alone is not enough)."""
    x1, y1, x2, y2 = box
    crop = gray[y1:y2, x1:x2]
    h, w = crop.shape
    if depth > 3 or h < 120 or w < 120:
        return [box]

    def part_score(b):
        r1, r2 = b[1] // cell, max(b[1] // cell + 1, b[3] // cell)
        c1, c2 = b[0] // cell, max(b[0] // cell + 1, b[2] // cell)
        return float(prob[r1:r2, c1:c2].mean())

    candidates = []
    for axis in (0, 1):                                     # 0 = horizontal cut (rows), 1 = vertical cut (cols)
        frac_line = (crop < 90).mean(axis=1 - axis)
        frac_white = (crop > 215).mean(axis=1 - axis)
        length = h if axis == 0 else w
        margin = int(length * 0.15)
        for kind, sep, min_run in (("rule", frac_line > 0.9, 2), ("gap", frac_white > 0.985, 4)):
            if kind == "gap" and prob is None:
                continue
            for a, b in _separator_runs(sep, min_run):
                c = (a + b) // 2
                if margin < c < length - margin:
                    candidates.append((kind == "rule", b - a, axis, c))
    for is_rule, _, axis, c in sorted(candidates, reverse=True):
        parts = ([[x1, y1, x2, y1 + c], [x1, y1 + c, x2, y2]] if axis == 0
                 else [[x1, y1, x1 + c, y2], [x1 + c, y1, x2, y2]])
        if not is_rule:
            # a white gap splits only two ad-looking parts that also look different from each other
            if not all(part_score(p) >= threshold * 0.9 for p in parts):
                continue
            if rgb is None or _palette_distance(rgb, parts[0], parts[1]) < 0.55:
                continue
        out = []
        for p in parts:
            out.extend(split_block(gray, p, prob, cell, threshold, depth + 1, rgb))
        return out
    return [box]


def detect_ad_boxes(work_arr: np.ndarray, prob: np.ndarray, cell: int, threshold: float = 0.5) -> list[dict]:
    """Ad boxes in work-image pixels, split where neighbouring ads touch, each with its mean ad likelihood."""
    import cv2

    gray = cv2.cvtColor(work_arr, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    rows, cols = prob.shape
    out = []
    for b in regions_from_map(prob, 1.0, cell, threshold):
        box = [int(b[0]), int(b[1]), min(W, int(b[2])), min(H, int(b[3]))]
        for p in split_block(gray, box, prob, cell, threshold, rgb=work_arr):
            r1, r2 = p[1] // cell, max(p[1] // cell + 1, p[3] // cell)
            c1, c2 = p[0] // cell, max(p[0] // cell + 1, p[2] // cell)
            score = float(prob[r1:r2, c1:c2].mean())
            area = (p[2] - p[0]) * (p[3] - p[1]) / (W * H)
            if area >= MIN_BLOCK_AREA * 0.8 and score >= threshold * 0.8:
                out.append({"box": p, "layout_score": score})
    # drop pieces inside a bigger block, and anything inside a (near) full-page ad
    as_ads = [{"bbox": {"x": o["box"][0], "y": o["box"][1], "w": o["box"][2] - o["box"][0], "h": o["box"][3] - o["box"][1]},
               "_o": o} for o in out]
    return [a["_o"] for a in _drop_nested(as_ads, W * H)]


# ---------------------------------------------------------------------------- reading each ad
AD_CUES = [r"\b\d{10}\b", r"\b\d{5}\s?\d{5}\b", r"\b1800[\s-]?\d{3}", r"(?:www\.|https?://|\.com\b|\.in\b|\.org\b)", r"@\w",
           r"₹\s?\d", r"\brs\.?\s?\d", r"\b\d+\s?%\s?(?:off|सूट|सवलत)", r"\b(?:offer|sale|discount|book now|call|contact|visit|"
           r"apply|admissions?|enrol|register|toll[- ]free|helpline|limited period|emi|interest|scheme|launch)\b",
           r"(?:संपर्क|सवलत|ऑफर|बुकिंग|नोंदणी|प्रवेश|योजना|व्याज|ठेव|आजच|फोन|कॉल|खरेदी|विक्री|उपलब्ध|मोफत|सेल)",
           r"(?:public notice|tender|e-auction|auction|notice is hereby|सूचना|निविदा|लिलाव|जाहीर)"]
NEWS_CUES = [r"लोकमत न्यूज नेटवर्क", r"(?:प्रतिनिधी|वृत्तसंस्था|वार्ताहर)\s*[:/]", r"\b(?:pti|ani|ians)\b",
             r"^(?:new delhi|mumbai|pune|nagpur|kolkata|chennai|bengaluru|hyderabad|thiruvananthapuram|washington)\s*[:,]",
             r"\b(?:said|told reporters|according to|officials said|the minister)\b",
             r"(?:नवी दिल्ली|मुंबई|पुणे|नागपूर)\s*:", r"(?:म्हणाले|सांगितले|असे ते म्हणाले|यांनी सांगितले)"]


def _enlarge_for_ocr(crop: Image.Image) -> Image.Image:
    w, h = crop.size
    target = min(OCR_MAX_WIDTH, max(OCR_MIN_WIDTH, w))
    if w < target:
        s = target / w
        crop = crop.resize((round(w * s), round(h * s)), Image.LANCZOS)
    elif w > OCR_MAX_WIDTH:
        s = OCR_MAX_WIDTH / w
        crop = crop.resize((round(w * s), round(h * s)), Image.LANCZOS)
    # gentle sharpening helps blurry phone photos and JPEG artefacts
    return crop.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))


def read_block(reader, crop: Image.Image) -> list[dict]:
    """OCR one ad crop -> lines with text, confidence, letter height and position (crop pixels)."""
    big = _enlarge_for_ocr(crop)
    res = reader.readtext(np.array(big), detail=1, paragraph=False, text_threshold=0.6, low_text=0.35,
                          link_threshold=0.4, contrast_ths=0.1, adjust_contrast=0.6, mag_ratio=1.0)
    lines = []
    for pts, text, conf in res:
        text = re.sub(r"\s+", " ", text or "").strip()
        if not text:
            continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        lines.append({"text": text, "conf": float(conf), "h": max(ys) - min(ys), "x": min(xs), "y": min(ys),
                      "w": max(xs) - min(xs)})
    return lines


def _clean_line(l: dict) -> bool:
    """Keep readable lines: words, and number-heavy lines like phone numbers and prices; drop punctuation junk."""
    t = l["text"].replace(" ", "")
    letters = sum(ch.isalpha() for ch in t)
    digits = sum(ch.isdigit() for ch in t)
    meaningful = letters + digits + t.count("₹") + t.count("%")
    return l["conf"] >= 0.3 and (letters >= 2 or digits >= 6) and meaningful >= 0.6 * max(1, len(t))


def compose_text(lines: list[dict]) -> dict:
    """Headline = biggest lettering; description = remaining readable lines in reading order."""
    good = [l for l in lines if _clean_line(l)]
    if not good:
        return {"headline": "", "description": "", "full_text": "", "lines": []}
    by_size = sorted(good, key=lambda l: (-l["h"], l["y"]))
    top_h = by_size[0]["h"]
    head = [l for l in by_size if l["h"] >= 0.75 * top_h][:3]
    head.sort(key=lambda l: (round(l["y"] / max(1, top_h)), l["x"]))
    headline = " ".join(l["text"] for l in head)
    rest = [l for l in good if l not in head]
    rest.sort(key=lambda l: (round(l["y"] / max(8, top_h * 0.5)), l["x"]))
    description = " ".join(l["text"] for l in rest)
    ordered = sorted(good, key=lambda l: (l["y"], l["x"]))
    return {"headline": headline[:160], "description": description[:600],
            "full_text": "\n".join(l["text"] for l in ordered), "lines": [l["text"] for l in ordered]}


def reading_quality(lines: list[dict]) -> float:
    """0..1: how confidently the text was read, weighted by text length, penalising garbage lines."""
    if not lines:
        return 0.0
    tot = sum(len(l["text"]) for l in lines) or 1
    conf = sum(l["conf"] * len(l["text"]) for l in lines) / tot
    clean = sum(len(l["text"]) for l in lines if _clean_line(l)) / tot
    return round(0.65 * conf + 0.35 * clean, 3)


def text_evidence(text: str) -> tuple[int, int]:
    t = text.lower()
    return (sum(bool(re.search(p, t, re.M)) for p in AD_CUES), sum(bool(re.search(p, t, re.M)) for p in NEWS_CUES))


# ---------------------------------------------------------------------------- main entry
_layout_model = None


def _layout():
    global _layout_model
    if _layout_model is None:
        import joblib
        from pathlib import Path
        _layout_model = joblib.load(Path(__file__).resolve().parent.parent / "models" / "layout_model.joblib")
    return _layout_model


def ad_probability_map(work: Image.Image):
    from .layout_features import cell_features
    m = _layout()
    F, shape = cell_features(np.array(work))
    clf = m["clf"]
    P = clf.predict_proba((F - m["mu"]) / m["sd"])[:, list(clf.classes_).index(1)]
    return P.reshape(shape), m["cell"]


def scan_page(image_bytes: bytes, debug: bool = False) -> dict:
    """Find and read every ad on a page image. Coordinates are in the uploaded image's pixels."""
    from .ocr import get_reader

    if not image_bytes:
        return {"ads": [], "total_ads": 0, "page_width": 0, "page_height": 0, "error": "No image provided"}
    try:
        img = load_image(image_bytes)
    except Exception:
        return {"ads": [], "total_ads": 0, "page_width": 0, "page_height": 0, "error": "This file is not a readable image."}
    work, s = _to_work(img)
    prob, cell = ad_probability_map(work)
    cands = detect_ad_boxes(np.array(work), prob, cell)
    reader = get_reader()
    ads, dbg = [], []
    for c in cands:
        x1, y1, x2, y2 = [v / s for v in c["box"]]
        x1, y1 = max(0, int(x1)), max(0, int(y1)); x2, y2 = min(img.width, int(x2)), min(img.height, int(y2))
        crop = img.crop((x1, y1, x2, y2))
        lines = read_block(reader, crop)
        text = compose_text(lines)
        ad_hits, news_hits = text_evidence(text["full_text"])
        score = c["layout_score"] + 0.08 * min(ad_hits, 3) - 0.15 * min(news_hits, 3)
        reasons = []
        if news_hits >= 2 and ad_hits == 0:
            score -= 0.3; reasons.append("news wording")
        if not text["headline"] and c["layout_score"] < 0.7:
            score -= 0.3; reasons.append("no readable text")
        area_frac = (x2 - x1) * (y2 - y1) / (img.width * img.height)
        if len(text["full_text"].replace(" ", "")) < 4 and area_frac < 0.08:
            score = 0; reasons.append("no readable text in a small block")
        is_ad = score >= 0.5
        dbg.append({"rect": [x1 * s, y1 * s, x2 * s, y2 * s], "score": score - 0.5, "is_ad": is_ad,
                    "source": f"L{c['layout_score']:.2f} a{ad_hits} n{news_hits}", "text": text["headline"][:60]})
        if not is_ad:
            continue
        q = reading_quality(lines)
        ads.append({**text, "bbox": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
                    "ocr_quality_score": q, "confidence": q,
                    "ad_likelihood": round(min(0.99, max(0.5, score)), 3)})
    ads = _drop_nested(ads, img.width * img.height)
    ads.sort(key=lambda a: (a["bbox"]["y"] // 40, a["bbox"]["x"]))
    for i, a in enumerate(ads):
        a["ad_index"] = i
    out = {"ads": ads, "total_ads": len(ads), "page_width": img.width, "page_height": img.height,
           "engine": "local", "error": None if ads else None}
    if debug:
        out["debug_blocks"] = dbg; out["scale"] = s
    return out



def _drop_nested(ads: list[dict], page_area: float) -> list[dict]:
    """Remove blocks that sit inside a bigger ad (fragments of one ad, or pieces of a full-page ad)."""
    def rect(a):
        b = a["bbox"]
        return (b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"])
    ads = sorted(ads, key=lambda a: -a["bbox"]["w"] * a["bbox"]["h"])
    kept = []
    for a in ads:
        ra = rect(a)
        area = max(1, a["bbox"]["w"] * a["bbox"]["h"])
        inside = any(_overlap(ra, rect(k)) >= 0.7 * area for k in kept)
        in_full_page = any(k["bbox"]["w"] * k["bbox"]["h"] >= 0.6 * page_area and _overlap(ra, rect(k)) >= 0.5 * area
                           for k in kept)
        if not (inside or in_full_page):
            kept.append(a)
    return kept
