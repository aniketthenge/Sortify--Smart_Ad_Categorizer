"""OCR helper module using EasyOCR for Marathi, Hindi, and English ad text extraction.

Performance notes:
- Images are resized to max 1000px before OCR to keep inference fast (~3-8s on CPU).
- The EasyOCR Reader is initialized once and cached (first call loads the model, ~10s).
- For full-page scans (multiple ads), images are resized to max 1600px for better
  detection and spatial clustering groups nearby text lines into distinct ad blocks.
"""
import io
import re
from PIL import Image, ImageDraw
import numpy as np

_reader = None

MAX_DIM = 800       # single-ad photo: resize to max 800px (~2-4s on CPU)
MAX_DIM_PAGE = 3600  # full-page scan: high resolution for better OCR accuracy and separating small text


def get_reader():
    """Initializes and caches the EasyOCR Reader for Marathi, Hindi, and English."""
    global _reader
    if _reader is None:
        import easyocr
        # 'mr' = Marathi, 'hi' = Hindi, 'en' = English
        _reader = easyocr.Reader(["mr", "hi", "en"], gpu=False, verbose=False)
    return _reader


def _resize(img: Image.Image, max_dim: int = MAX_DIM) -> Image.Image:
    """Shrink large images so OCR doesn't take forever on high-res phone photos."""
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def extract_text_from_bytes(image_bytes: bytes) -> dict:
    """Extracts text from ad image bytes using EasyOCR.
    Returns:
        dict: {"headline": str, "description": str, "full_text": str, "lines": list[str]}
    """
    if not image_bytes:
        return {"headline": "", "description": "", "full_text": "", "lines": []}
    try:
        reader = get_reader()
        img = _resize(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        # EasyOCR expects a numpy array, not a PIL Image
        results = reader.readtext(np.array(img), detail=0)
        lines = [re.sub(r"\s+", " ", r).strip() for r in results if r and r.strip()]
        if not lines:
            return {"headline": "", "description": "", "full_text": "", "lines": []}

        headline = lines[0]
        description = " ".join(lines[1:]) if len(lines) > 1 else ""
        return {
            "headline": headline,
            "description": description,
            "full_text": "\n".join(lines),
            "lines": lines,
        }
    except Exception as e:
        return {"headline": "", "description": "", "full_text": "", "lines": [], "error": str(e)}


# ---------------------------------------------------------------------------
# Multi-ad page scanning: detect several ads in one image / newspaper page.
# ---------------------------------------------------------------------------

def _bbox_center(bbox):
    """Return (cx, cy) of an EasyOCR bounding-box polygon (list of 4 [x,y] points)."""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def _bbox_rect(bbox):
    """Return (x1, y1, x2, y2) bounding rectangle from EasyOCR polygon."""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def _cluster_text_blocks(detections: list, gap_ratio: float = 0.02, img_height: int = 800) -> list[list]:
    """Group detected text lines into spatial clusters (ad blocks).

    Two text lines belong to the same ad if their vertical distance is small
    relative to the image height and they overlap horizontally. This works well
    for newspaper / magazine page layouts where ads are rectangular blocks of
    text separated by whitespace or rules.

    Args:
        detections: EasyOCR results with detail=1 — list of (bbox, text, conf).
        gap_ratio: max vertical gap (as fraction of image height) to merge lines.
        img_height: height of the (resized) image in pixels.

    Returns:
        List of clusters, each cluster is a list of (bbox, text, conf) sorted
        top-to-bottom.
    """
    if not detections:
        return []

    max_gap = max(img_height * gap_ratio, 30)  # at least 30px

    # Sort all detections by vertical center
    items = sorted(detections, key=lambda d: _bbox_center(d[0])[1])
    clusters: list[list] = [[items[0]]]

    for det in items[1:]:
        merged = False
        rect = _bbox_rect(det[0])
        cy = _bbox_center(det[0])[1]

        for cluster in clusters:
            # Check against every line in this cluster for proximity
            for member in cluster:
                m_rect = _bbox_rect(member[0])
                m_cy = _bbox_center(member[0])[1]

                # Vertical proximity: centres are within max_gap
                if abs(cy - m_cy) > max_gap * 2.5:
                    continue

                # Horizontal overlap: the two text regions share some x-range
                x_overlap = min(rect[2], m_rect[2]) - max(rect[0], m_rect[0])
                min_width = min(rect[2] - rect[0], m_rect[2] - m_rect[0])
                if min_width > 0 and x_overlap > -min_width * 0.3:
                    cluster.append(det)
                    merged = True
                    break
            if merged:
                break

        if not merged:
            clusters.append([det])

    # Sort each cluster's lines top-to-bottom
    for cluster in clusters:
        cluster.sort(key=lambda d: _bbox_center(d[0])[1])

    return clusters


def _cluster_to_ad(cluster: list, ad_index: int) -> dict:
    """Convert a spatial cluster of OCR detections into an ad dict.

    Returns:
        dict with keys: ad_index, headline, description, full_text, lines,
        confidence (avg OCR confidence), bbox (bounding rect of the whole cluster).
    """
    lines = [re.sub(r"\s+", " ", det[1]).strip() for det in cluster if det[1] and det[1].strip()]
    confs = [det[2] for det in cluster if det[1] and det[1].strip()]
    if not lines:
        return None

    # Bounding rect enclosing all text in the cluster
    all_rects = [_bbox_rect(det[0]) for det in cluster]
    x1 = min(r[0] for r in all_rects)
    y1 = min(r[1] for r in all_rects)
    x2 = max(r[2] for r in all_rects)
    y2 = max(r[3] for r in all_rects)

    # Heuristic: the first (topmost) line is the headline
    headline = lines[0]
    description = " ".join(lines[1:]) if len(lines) > 1 else ""

    return {
        "ad_index": ad_index,
        "headline": headline,
        "description": description,
        "full_text": "\n".join(lines),
        "lines": lines,
        "confidence": round(sum(confs) / len(confs), 3) if confs else 0,
        "bbox": {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)},
    }


def is_ad(text: str) -> bool:
    """Perfectly differentiate between a news article and an advertisement."""
    text_lower = text.lower()
    
    # Strong Ad indicators
    ad_patterns = [
        r'\b\d{10}\b', r'\b\d{5}\s\d{5}\b', r'\bph[:\.]?\s*\d+', r'\bmob[:\.]?\s*\d+', r'contact\s*(us|at)?[:\.]?',
        r'\bwanted\b', r'\brequired\b', r'\bfor sale\b', r'\bto let\b', r'\brent\b', r'\bbhk\b', r'\boffers?\b',
        r'\bdiscount\b', r'\b₹\d+', r'rs\.?\s*\d+', r'\bwalk-in\b', r'\bvacancy\b', r'\bsituation vacant\b',
        r'\bmatrimonial\b', r'\bbride\b', r'\bgroom\b', r'\bname change\b', r'\bpublic notice\b', r'\btender\b',
        r'\badmission\b', r'\bapply now\b', r'\blakhs?\b', r'\bcrores?\b', r'www\.', r'\.com\b', r'@'
    ]
    
    # Strong News indicators
    news_patterns = [
        r'\b[a-z\s]+(correspondent|reporter)\b',
        r'\b(said|told|added|announced|according to)\b',
        r'\b(police|government|minister|chief minister|pm|president)\b',
        r'\bnew delhi[:\s]', r'\bmumbai[:\s]', r'\bpune[:\s]', r'\bnagpur[:\s]',
        r'\bpti\b', r'\bpress trust\b', r'\bnews network\b', r'\beditorial\b'
    ]
    
    ad_score = sum(1 for p in ad_patterns if re.search(p, text_lower))
    news_score = sum(1 for p in news_patterns if re.search(p, text_lower))
    
    # News content typically has many journalistic keywords and very few ad keywords.
    if news_score > ad_score and ad_score <= 1:
        return False
    if news_score >= 3 and ad_score <= 2:
        return False
        
    return True


def extract_multiple_ads_from_bytes(image_bytes: bytes, min_lines: int = 1) -> dict:
    """Scan a page/image for multiple ads, cluster them spatially, extract text.

    Args:
        image_bytes: raw image file bytes (JPEG, PNG, WebP).
        min_lines: minimum number of text lines for a cluster to count as an ad.

    Returns:
        dict: {
            "ads": list[dict],        # each ad has headline, description, bbox, etc.
            "total_ads": int,
            "page_width": int,
            "page_height": int,
            "error": str | None,
        }
    """
    if not image_bytes:
        return {"ads": [], "total_ads": 0, "page_width": 0, "page_height": 0, "error": "No image provided"}

    try:
        reader = get_reader()
        img = _resize(Image.open(io.BytesIO(image_bytes)).convert("RGB"), max_dim=MAX_DIM_PAGE)
        page_w, page_h = img.size
        arr = np.array(img)

        # detail=1 gives bounding boxes: list of (bbox, text, conf)
        detections = reader.readtext(arr, detail=1)
        # Filter out very low-confidence noise
        detections = [d for d in detections if d[2] > 0.15 and d[1] and d[1].strip()]

        if not detections:
            return {"ads": [], "total_ads": 0, "page_width": page_w, "page_height": page_h,
                    "error": "No text detected on this page."}

        clusters = _cluster_text_blocks(detections, img_height=page_h)

        ads = []
        for i, cluster in enumerate(clusters):
            if len(cluster) < min_lines:
                continue
            ad = _cluster_to_ad(cluster, i)
            if ad and is_ad(ad["full_text"]):
                ads.append(ad)

        # Sort ads top-to-bottom, left-to-right
        ads.sort(key=lambda a: (a["bbox"]["y"], a["bbox"]["x"]))
        # Re-index after sorting
        for i, ad in enumerate(ads):
            ad["ad_index"] = i

        return {
            "ads": ads,
            "total_ads": len(ads),
            "page_width": page_w,
            "page_height": page_h,
            "error": None,
        }
    except Exception as e:
        return {"ads": [], "total_ads": 0, "page_width": 0, "page_height": 0, "error": str(e)}
