"""OCR helper module using EasyOCR for Marathi, Hindi, and English ad text extraction.

Performance notes:
- Images are resized to max 1000px before OCR to keep inference fast (~3-8s on CPU).
- The EasyOCR Reader is initialized once and cached (first call loads the model, ~10s).
"""
import io
import re
from PIL import Image
import numpy as np

_reader = None

MAX_DIM = 800  # resize photos to max 800px so OCR runs in ~2-4s on CPU


def get_reader():
    """Initializes and caches the EasyOCR Reader for Marathi, Hindi, and English."""
    global _reader
    if _reader is None:
        import easyocr
        # 'mr' = Marathi, 'hi' = Hindi, 'en' = English
        _reader = easyocr.Reader(["mr", "hi", "en"], gpu=False, verbose=False)
    return _reader


def _resize(img: Image.Image) -> Image.Image:
    """Shrink large images so OCR doesn't take forever on high-res phone photos."""
    w, h = img.size
    if max(w, h) <= MAX_DIM:
        return img
    scale = MAX_DIM / max(w, h)
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
