"""Train the page-layout model (ad area vs news area) from test_scans/ground_truth.json.

    .venv\\Scripts\\python tools\\train_layout.py --cv     # honest test: train on all pages but one, test on that one
    .venv\\Scripts\\python tools\\train_layout.py --fit    # train on every page -> models/layout_model.joblib
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import page_scan as ps  # noqa: E402
from app.layout_features import CELL, cell_features  # noqa: E402
from app.softmax import SoftmaxRegression  # noqa: E402

SCANS = ROOT / "test_scans"


def page_data(name, gt, variant="orig"):
    from tools.eval_scan import degrade
    img = ps.load_image((SCANS / name).read_bytes())
    img, k = degrade(img, variant)
    work, s = ps._to_work(img)
    s = s * k                     # coordinates in the answer key are for the original image
    F, (rows, cols) = cell_features(np.array(work))
    cy = (np.arange(rows) + 0.5) * CELL / s
    cx = (np.arange(cols) + 0.5) * CELL / s
    yy, xx = np.meshgrid(cy, cx, indexing="ij")
    y = np.zeros((rows, cols), np.int8); w = np.ones((rows, cols), np.float32)
    for g in gt["ads"]:
        x1, y1, x2, y2 = g["box"]; y[(xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)] = 1
    for o in gt["optional"]:
        x1, y1, x2, y2 = o["box"]; w[(xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2) & (y == 0)] = 0
    return dict(name=name, variant=variant, F=F, y=y.ravel(), w=w.ravel(), shape=(rows, cols), scale=s, size=img.size)


def fit(pages, C=1.0):
    X = np.concatenate([p["F"] for p in pages]); y = np.concatenate([p["y"] for p in pages])
    w = np.concatenate([p["w"] for p in pages]); keep = w > 0
    X, y = X[keep], y[keep]
    if len(y) > 40000:   # a random sample trains ~4x faster with the same result
        idx = np.random.default_rng(0).choice(len(y), 40000, replace=False)
        X, y = X[idx], y[idx]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = SoftmaxRegression(C=C).fit((X - mu) / sd, y)
    return {"mu": mu, "sd": sd, "clf": clf, "cell": CELL}


def _draw(p, prob, boxes, g):
    from PIL import Image, ImageDraw
    img = ps.load_image((SCANS / p["name"]).read_bytes()).convert("RGB")
    heat = Image.fromarray((np.clip(prob, 0, 1) * 255).astype(np.uint8)).resize(img.size, Image.NEAREST)
    red = Image.new("RGB", img.size, (255, 0, 0))
    img = Image.composite(red, img, heat.point(lambda v: int(v * 0.45)))
    d = ImageDraw.Draw(img)
    for ad in g["ads"]: d.rectangle(ad["box"], outline=(0, 90, 255), width=4)
    for b in boxes: d.rectangle(b, outline=(0, 170, 0), width=3)
    (SCANS / "debug").mkdir(exist_ok=True)
    img.save(SCANS / "debug" / ("cv_" + Path(p["name"]).stem + ".jpg"), quality=80)


def predict_map(model, p):
    P = model["clf"].predict_proba((p["F"] - model["mu"]) / model["sd"])[:, list(model["clf"].classes_).index(1)]
    return P.reshape(p["shape"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true"); ap.add_argument("--fit", action="store_true")
    ap.add_argument("--C", type=float, default=0.05)
    a = ap.parse_args()
    gt = json.loads((SCANS / "ground_truth.json").read_text(encoding="utf-8"))["pages"]
    variants = ("orig", "low", "photo")
    pages = [page_data(n, g, v) for n, g in gt.items() for v in variants]   # augmented: 3 quality levels
    print("cells:", sum(len(p["y"]) for p in pages), "ad cells:", int(sum(p["y"].sum() for p in pages)), flush=True)
    if a.cv:
        from tools.eval_scan import matches
        tot_found = tot_ads = tot_fp = 0
        by_var = {}
        for name in gt:
            model = fit([q for q in pages if q["name"] != name], a.C)      # this page unseen in any quality
            for p in [q for q in pages if q["name"] == name]:
                prob = predict_map(model, p)
                from tools.eval_scan import degrade
                img, k = degrade(ps.load_image((SCANS / name).read_bytes()), p["variant"])
                work, s_ = ps._to_work(img)
                boxes = [[v / p["scale"] for v in d["box"]] for d in ps.detect_ad_boxes(np.array(work), prob, CELL)]
                g = gt[name]
                used = set(); found = 0
                for ad in g["ads"]:
                    hit = next((k for k, b in enumerate(boxes) if k not in used and matches(b, ad["box"])), None)
                    if hit is not None: used.add(hit); found += 1
                fp = sum(1 for k, b in enumerate(boxes) if k not in used and not any(matches(b, o["box"]) for o in g["optional"] + g["ads"]))
                tot_found += found; tot_ads += len(g["ads"]); tot_fp += fp
                v = by_var.setdefault(p["variant"], [0, 0, 0]); v[0] += found; v[1] += len(g["ads"]); v[2] += fp
                if p["variant"] == "orig": _draw(p, prob, boxes, g)
                print(f"  {name:30s} {p['variant']:5s} ads found {found}/{len(g['ads'])}  extra blocks {fp}", flush=True)
        for k, v in by_var.items(): print(f"   {k:5s}: ads found {v[0]}/{v[1]}, extra blocks {v[2]}")
        print(f"== held-out pages: ads found {tot_found}/{tot_ads}, extra blocks {tot_fp}")
    if a.fit:
        model = fit(pages, a.C)
        import joblib
        joblib.dump(model, ROOT / "models" / "layout_model.joblib")
        print("saved models/layout_model.joblib")
