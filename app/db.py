"""SQLite persistence for ads, scrape runs and human labels."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "ads.db"
SAMPLE_PATH = ROOT / "data" / "sample_ads.json"

SCHEMA = """
CREATE TABLE IF NOT EXISTS ads (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    description   TEXT DEFAULT '',
    advertiser    TEXT DEFAULT '',
    landing_url   TEXT DEFAULT '',
    domain        TEXT DEFAULT '',
    image_url     TEXT DEFAULT '',
    network       TEXT DEFAULT '',
    source_page   TEXT DEFAULT '',
    first_seen    TEXT,
    last_seen     TEXT,
    times_seen    INTEGER DEFAULT 1,
    category      TEXT,
    confidence    REAL,
    method        TEXT,
    needs_review  INTEGER DEFAULT 0,
    manual_label  INTEGER DEFAULT 0      -- 1 = category set by a human
);
CREATE INDEX IF NOT EXISTS idx_ads_category ON ads(category);
CREATE TABLE IF NOT EXISTS scrape_runs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT,
    finished_at TEXT,
    status     TEXT,
    ads_found  INTEGER DEFAULT 0,
    new_ads    INTEGER DEFAULT 0,
    log        TEXT DEFAULT ''
);
"""


@contextmanager
def connect(path: Path = DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: Path = DB_PATH):
    with connect(path) as c:
        c.executescript(SCHEMA)


def upsert_ads(ads: list[dict], classifier, path: Path = DB_PATH) -> int:
    """Insert new ads (classified on the way in) or bump last_seen/times_seen.
    Human labels are never overwritten. Returns the number of new ads."""
    with connect(path) as c:
        fresh = []
        for ad in ads:
            if c.execute("SELECT 1 FROM ads WHERE id = ?", (ad["id"],)).fetchone():
                c.execute("UPDATE ads SET last_seen = ?, times_seen = times_seen + 1 WHERE id = ?",
                          (ad.get("scraped_at"), ad["id"]))
            elif ad["id"] not in {a["id"] for a in fresh}:
                fresh.append(ad)
        for ad, pred in zip(fresh, classifier.predict_many(fresh)):
            c.execute(
                """INSERT INTO ads (id, title, description, advertiser, landing_url, domain, image_url, network,
                   source_page, first_seen, last_seen, category, confidence, method, needs_review)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ad["id"], ad["title"], ad.get("description", ""), ad.get("advertiser", ""), ad.get("landing_url", ""),
                 ad.get("domain", ""), ad.get("image_url", ""), ad.get("network", ""), ad.get("source_page", ""),
                 ad.get("scraped_at"), ad.get("scraped_at"), pred["category"], pred["confidence"], pred["method"],
                 int(pred["needs_review"])))
    return len(fresh)


def reclassify_all(classifier, path: Path = DB_PATH) -> int:
    """Re-run the (retrained) model over every ad that has no human label."""
    with connect(path) as c:
        rows = [dict(r) for r in c.execute(
            "SELECT id, title, description, advertiser, domain FROM ads WHERE manual_label = 0")]
        for r, p in zip(rows, classifier.predict_many(rows)):
            c.execute("UPDATE ads SET category=?, confidence=?, method=?, needs_review=? WHERE id=?",
                      (p["category"], p["confidence"], p["method"], int(p["needs_review"]), r["id"]))
    return len(rows)


def set_label(ad_id: str, category: str, path: Path = DB_PATH) -> dict | None:
    """Store a staff correction. Returns the ad (advertiser/domain) or None if not found."""
    with connect(path) as c:
        cur = c.execute("UPDATE ads SET category=?, confidence=1.0, method='manual', needs_review=0, manual_label=1 "
                        "WHERE id=?", (category, ad_id))
        if not cur.rowcount:
            return None
        return dict(c.execute("SELECT advertiser, domain FROM ads WHERE id=?", (ad_id,)).fetchone())


def categories_for(ids: list[str], path: Path = DB_PATH) -> dict[str, str]:
    if not ids:
        return {}
    with connect(path) as c:
        q = f"SELECT id, category FROM ads WHERE id IN ({','.join('?' * len(ids))})"
        return {r["id"]: r["category"] for r in c.execute(q, ids)}


def feedback_rows(path: Path = DB_PATH) -> list[dict]:
    """Human-labelled ads, fed back into training."""
    with connect(path) as c:
        rows = c.execute("SELECT title, description, advertiser, domain, category FROM ads WHERE manual_label = 1")
        return [dict(r) for r in rows]


def query_ads(category: str = "", q: str = "", review_only: bool = False, network: str = "",
              path: Path = DB_PATH) -> list[dict]:
    sql, args = "SELECT * FROM ads WHERE 1=1", []
    if category:
        sql += " AND category = ?"; args.append(category)
    if network:
        sql += " AND network = ?"; args.append(network)
    if review_only:
        sql += " AND needs_review = 1 AND manual_label = 0"
    if q:
        sql += " AND (title LIKE ? OR description LIKE ? OR advertiser LIKE ? OR domain LIKE ?)"
        args += [f"%{q}%"] * 4
    sql += " ORDER BY last_seen DESC, times_seen DESC"
    with connect(path) as c:
        return [dict(r) for r in c.execute(sql, args)]


def stats(path: Path = DB_PATH) -> dict:
    with connect(path) as c:
        one = lambda sql: c.execute(sql).fetchone()[0]
        return {
            "total_ads": one("SELECT COUNT(*) FROM ads"),
            "impressions": one("SELECT COALESCE(SUM(times_seen),0) FROM ads"),
            "advertisers": one("SELECT COUNT(DISTINCT domain) FROM ads"),
            "needs_review": one("SELECT COUNT(*) FROM ads WHERE needs_review=1 AND manual_label=0"),
            "manual": one("SELECT COUNT(*) FROM ads WHERE manual_label=1"),
            "avg_confidence": one("SELECT ROUND(AVG(confidence),3) FROM ads") or 0,
            "last_updated": _fmt_date(one("SELECT MAX(last_seen) FROM ads")),
            "by_category": [dict(r) for r in c.execute(
                "SELECT category, COUNT(*) AS n, SUM(times_seen) AS impressions, ROUND(AVG(confidence),2) AS conf "
                "FROM ads GROUP BY category ORDER BY n DESC")],
            "by_network": [dict(r) for r in c.execute("SELECT network, COUNT(*) AS n FROM ads GROUP BY network ORDER BY n DESC")],
            "top_advertisers": [dict(r) for r in c.execute(
                "SELECT advertiser, domain, category, COUNT(*) AS creatives, SUM(times_seen) AS impressions "
                "FROM ads GROUP BY LOWER(advertiser), domain ORDER BY creatives DESC, impressions DESC LIMIT 80")],
            "runs": [dict(r) for r in c.execute("SELECT id, started_at, finished_at, status, ads_found, new_ads "
                                                 "FROM scrape_runs ORDER BY id DESC LIMIT 5")],
        }


def _fmt_date(iso: str | None) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %Y") if iso else ""
    except ValueError:
        return ""


def start_run(path: Path = DB_PATH) -> int:
    with connect(path) as c:
        return c.execute("INSERT INTO scrape_runs (started_at, status) VALUES (?, 'running')",
                         (datetime.now().isoformat(timespec="seconds"),)).lastrowid


def finish_run(run_id: int, status: str, found: int, new: int, log: str, path: Path = DB_PATH):
    with connect(path) as c:
        c.execute("UPDATE scrape_runs SET finished_at=?, status=?, ads_found=?, new_ads=?, log=? WHERE id=?",
                  (datetime.now().isoformat(timespec="seconds"), status, found, new, log, run_id))


def load_sample_if_empty(classifier, path: Path = DB_PATH) -> int:
    """First launch: seed the DB with the ads scraped while building this project,
    so the dashboard is populated even before the user runs a live scrape."""
    with connect(path) as c:
        if c.execute("SELECT COUNT(*) FROM ads").fetchone()[0]:
            return 0
    if not SAMPLE_PATH.exists():
        return 0
    ads = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    return upsert_ads(ads, classifier, path)


def last_successful_run(path: Path = DB_PATH) -> datetime | None:
    """When ads were last refreshed successfully (None if never)."""
    with connect(path) as c:
        row = c.execute("SELECT MAX(finished_at) FROM scrape_runs WHERE status='success'").fetchone()
    try:
        return datetime.fromisoformat(row[0]) if row and row[0] else None
    except ValueError:
        return None
