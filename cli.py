"""
Command-line interface.

  python cli.py scrape [--sections / /business/ ...]   scrape live ads into the DB
  python cli.py train                                    retrain (includes your UI corrections)
  python cli.py classify "headline" [--advertiser X] [--domain y.com]
  python cli.py stats                                    category summary of stored ads
  python cli.py export out.csv                           dump all ads to CSV
"""
import argparse
import csv
import json
import sys

from app import db
from app.classifier import AdClassifier

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Marathi text on the Windows console


def main():
    ap = argparse.ArgumentParser(description="Lokmat Times ad categorizer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scrape"); s.add_argument("--sections", nargs="*"); s.add_argument("--show-browser", action="store_true")
    sub.add_parser("train")
    c = sub.add_parser("classify"); c.add_argument("title"); c.add_argument("--description", default="")
    c.add_argument("--advertiser", default=""); c.add_argument("--domain", default="")
    sub.add_parser("stats")
    e = sub.add_parser("export"); e.add_argument("path")
    args = ap.parse_args()

    db.init_db()
    if args.cmd == "scrape":
        from app.scraper import scrape
        model = AdClassifier.load()
        run = db.start_run()
        ads = scrape(args.sections, headless=not args.show_browser)
        new = db.upsert_ads(ads, model)
        db.finish_run(run, "success", len(ads), new, "cli")
        print(f"{len(ads)} unique ads scraped, {new} new.")
    elif args.cmd == "train":
        model = AdClassifier.train(extra_rows=db.feedback_rows())
        ev = model.meta["evaluation"]
        print(f"Trained on {model.meta['n_samples']} examples. CV accuracy {ev['accuracy']:.1%}, macro-F1 {ev['macro_f1']:.2f}")
        print(f"Re-classified {db.reclassify_all(model)} stored ads.")
    elif args.cmd == "classify":
        r = AdClassifier.load().predict(args.title, args.description, args.advertiser, args.domain)
        print(json.dumps(r, indent=2, ensure_ascii=False))
    elif args.cmd == "stats":
        st = db.stats()
        print(f"{st['total_ads']} ads, {st['advertisers']} advertisers, {st['needs_review']} need review\n")
        for row in st["by_category"]:
            print(f"  {row['category']:<32} {row['n']:>4}  (avg conf {row['conf']})")
    elif args.cmd == "export":
        rows = db.query_ads()
        with open(args.path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
            w.writeheader(); w.writerows(rows)
        print(f"Wrote {len(rows)} ads to {args.path}")


if __name__ == "__main__":
    main()
