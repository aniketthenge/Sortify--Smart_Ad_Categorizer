"""Score the full-page ad scanner against the hand-marked answer key (test_scans/ground_truth.json).

    .venv\\Scripts\\python tools\\eval_scan.py                 # current scanner, original pages
    .venv\\Scripts\\python tools\\eval_scan.py --variants      # also low-quality and phone-photo copies
    .venv\\Scripts\\python tools\\eval_scan.py --engine legacy # the old scanner, for comparison

Reports: ads found (recall), non-ads reported (false positives), category accuracy on found ads,
average reading (OCR) quality and category confidence.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SCANS = ROOT / "test_scans"


def degrade(img: Image.Image, kind: str) -> tuple[Image.Image, float]:
    """Return a degraded copy and the scale factor applied to coordinates."""
    if kind == "low":  # small, heavily compressed (bad WhatsApp forward)
        s = 0.6
        small = img.resize((int(img.width * s), int(img.height * s)), Image.BILINEAR)
        buf = io.BytesIO(); small.convert("RGB").save(buf, "JPEG", quality=35)
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), s
    if kind == "photo":  # slightly blurred and tilted phone photo
        out = img.convert("RGB").filter(ImageFilter.GaussianBlur(1.1)).rotate(1.5, expand=False, fillcolor=(235, 235, 230))
        return out, 1.0
    return img.convert("RGB"), 1.0


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0


def matches(det, gt):
    """A detection counts for an answer-key box if they overlap well, or the detection sits inside it
    and covers a fair part of it."""
    cx, cy = (det[0] + det[2]) / 2, (det[1] + det[3]) / 2
    inside = gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]
    area_d = (det[2] - det[0]) * (det[3] - det[1]); area_g = (gt[2] - gt[0]) * (gt[3] - gt[1])
    return iou(det, gt) >= 0.3 or (inside and area_d >= 0.25 * area_g)


def run(engine: str, variants: list[str], verbose: bool, cv: bool = False):
    from app.classifier import AdClassifier
    if engine == "legacy":
        from app.ocr import extract_multiple_ads_from_bytes as scan_fn
    else:
        from app.page_scan import scan_page as scan_fn
    model = AdClassifier.load()
    gt_all = json.loads((SCANS / "ground_truth.json").read_text(encoding="utf-8"))["pages"]
    layout_for = None
    if cv and engine == "new":   # honest test: the layout model never saw the page it is scanning
        from app import page_scan
        from tools import train_layout as tl
        tr_pages = [tl.page_data(n, g, v) for n, g in gt_all.items() for v in ("orig", "low", "photo")]
        cache = {}
        def layout_for(name):
            if name not in cache:
                cache.clear(); cache[name] = tl.fit([q for q in tr_pages if q["name"] != name])
            page_scan._layout_model = cache[name]
    totals = {}
    for kind in variants:
        found = n_ads = fp = cat_ok = 0
        ocr_q, conf = [], []
        t0 = time.time()
        for name, gt in gt_all.items():
            if layout_for:
                layout_for(name)
            img, s = degrade(Image.open(SCANS / name), kind)
            buf = io.BytesIO(); img.save(buf, "PNG")
            res = scan_fn(buf.getvalue())
            dets = []
            for ad in res.get("ads", []):
                b = ad["bbox"]; W = res.get("page_width") or img.width; scale = img.width / W
                box = [b["x"] * scale / s, b["y"] * scale / s, (b["x"] + b["w"]) * scale / s, (b["y"] + b["h"]) * scale / s]
                cat = ad.get("category")
                if not cat:
                    p = model.predict(ad.get("headline", ""), ad.get("description", ""), ad.get("advertiser", ""))
                    cat, c = p["category"], p["confidence"]
                else:
                    c = ad.get("category_confidence", 0)
                dets.append({"box": box, "cat": cat, "conf": c, "ocr": ad.get("ocr_quality_score", ad.get("confidence", 0)),
                             "text": (ad.get("headline", "") + " | " + ad.get("description", ""))[:70]})
            used = set()
            for g in gt["ads"]:
                n_ads += 1
                hit = next((i for i, d in enumerate(dets) if i not in used and matches(d["box"], g["box"])), None)
                if hit is not None:
                    used.add(hit); found += 1
                    d = dets[hit]; ok = d["cat"] == g["category"] or d["cat"] in g.get("also_ok", [])
                    cat_ok += ok; ocr_q.append(d["ocr"]); conf.append(d["conf"])
                    if verbose: print(f"  [{kind}] {name}: FOUND {g['label'][:28]:28s} -> {d['cat'][:22]:22s} {'OK ' if ok else 'BAD'} conf={d['conf']:.2f} ocr={d['ocr']:.2f} | {d['text']}")
                elif verbose:
                    print(f"  [{kind}] {name}: MISSED {g['label']}")
            for i, d in enumerate(dets):
                if i in used:
                    continue
                if any(matches(d["box"], o["box"]) for o in gt["optional"] + gt["ads"]):
                    continue
                fp += 1
                if verbose: print(f"  [{kind}] {name}: FALSE {d['cat'][:22]:22s} | {d['text']}")
        avg = lambda xs: sum(xs) / len(xs) if xs else 0
        totals[kind] = dict(found=found, of=n_ads, false_pos=fp, cat_ok=cat_ok, ocr=round(avg(ocr_q), 2),
                            conf=round(avg(conf), 2), secs=round(time.time() - t0))
        print(f"== {engine} / {kind}: ads found {found}/{n_ads} · non-ads reported {fp} · right category {cat_ok}/{found}"
              f" · avg reading quality {avg(ocr_q):.2f} · avg confidence {avg(conf):.2f} · {time.time() - t0:.0f}s", flush=True)
    return totals


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="new", choices=["new", "legacy"])
    ap.add_argument("--variants", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--cv", action="store_true", help="retrain the layout model without each page before scanning it")
    a = ap.parse_args()
    run(a.engine, ["orig", "low", "photo"] if a.variants else ["orig"], a.verbose, a.cv)
