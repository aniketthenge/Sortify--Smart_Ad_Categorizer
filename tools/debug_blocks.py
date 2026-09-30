"""Draw the scanner's candidate blocks (and final ads, with --full) on each test page -> test_scans/debug/.

    .venv\\Scripts\\python tools\\debug_blocks.py           # layout candidates only (fast)
    .venv\\Scripts\\python tools\\debug_blocks.py --full    # full scan: green = ad, red = rejected
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import page_scan as ps  # noqa: E402

SCANS = ROOT / "test_scans"
OUT = SCANS / "debug"
OUT.mkdir(exist_ok=True)
gt = json.loads((SCANS / "ground_truth.json").read_text(encoding="utf-8"))["pages"]
full = "--full" in sys.argv
only = [a for a in sys.argv[1:] if not a.startswith("--")]

for name in gt:
    if only and not any(o in name for o in only):
        continue
    img = ps.load_image((SCANS / name).read_bytes())
    work, s = ps._to_work(img)
    d = ImageDraw.Draw(work)
    for g in gt[name]["ads"]:
        b = [v * s for v in g["box"]]
        d.rectangle(b, outline=(0, 90, 255), width=7)
    if full:
        res = ps.scan_page((SCANS / name).read_bytes(), debug=True)
        for blk in res["debug_blocks"]:
            col = (0, 170, 0) if blk["is_ad"] else (220, 0, 0)
            r = [v * s / res["scale"] for v in blk["rect"]]
            d.rectangle(r, outline=col, width=4)
            d.text((r[0] + 6, r[1] + 6), f"{blk['score']:+.1f} {blk['source']}", fill=col)
    else:
        blocks, page = ps.find_candidate_blocks(work)
        for b in blocks:
            col = (255, 140, 0) if b.source == "box" else (200, 0, 200)
            d.rectangle(b.rect(), outline=col, width=4)
            d.text((b.x1 + 6, b.y1 + 6), f"{b.source} c={b.color_frac:.2f}", fill=col)
        print(f"{name}: {len(blocks)} candidates, page colour {page.color_frac:.2f}")
    work.thumbnail((900, 900))
    work.save(OUT / (Path(name).stem + (".full" if full else "") + ".jpg"), quality=80)
