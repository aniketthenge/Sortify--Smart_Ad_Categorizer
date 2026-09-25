"""Flask web application.

Customer pages : Overview (/), Ad Library (/ads), Live Site (/live), Categorize an Ad (/classify).
Staff page     : /admin  - protected (see _staff_allowed), never linked from customer pages.

Customer-facing responses (HTML, JS, JSON) carry no information about how ads are
collected or how the model works; tests/test_app.py enforces this.
"""
from __future__ import annotations

import csv
import hmac
import io
import os
import re
import threading
import time
from datetime import datetime, timedelta

from flask import Flask, Response, abort, jsonify, render_template, request, send_file, send_from_directory

from . import db
from .classifier import CATEGORY_NAMES, AdClassifier, category_colors
from .scraper import clean_advertiser, strip_tracking

LIVE_SECTIONS = {"Home": "/", "Chhatrapati Sambhajinagar": "/chhatrapati-sambhajinagar/", "Pune": "/pune/", "National": "/national/",
                 "Business": "/business/", "Entertainment": "/entertainment/", "Lifestyle": "/lifestyle/",
                 "Sports": "/sports/", "Technology": "/technology/"}
CATEGORY_ICONS = {
    "Finance & Investment": "💰", "Insurance": "🛡️", "Health & Wellness": "🩺", "Matrimony & Dating": "💍",
    "Gaming": "🎮", "Education & Courses": "🎓", "Real Estate": "🏠", "Home & Living": "🛋️",
    "Fashion & Shopping": "👗", "Technology & Electronics": "💻", "Automotive": "🚗", "Travel & Tourism": "✈️",
    "Charity & NGO": "🤝", "Astrology & Spirituality": "🔮", "Industrial & B2B": "🏭", "Food & Beverage": "🍲",
    "Entertainment & Media": "🎬", "Professional Services": "⚖️", "Jobs & Careers": "💼",
    "Government & Public Notices": "🏛️", "Sponsored Stories": "📰",
}
PAGE_SIZE = 24
NETWORK_DOMAINS = {"srchly.co", "searchinfohub.com", "searchresultoffers.com"}
PUBLIC_AD_FIELDS = ("id", "title", "description", "advertiser", "domain", "landing_url", "category",
                    "first_seen", "last_seen")

_model_lock = threading.Lock()
_refresh_state = {"running": False, "log": [], "run_id": None, "result": None}
_refresh_lock = threading.Lock()


def create_app(db_path=None, auto_refresh: bool | None = None) -> Flask:
    app = Flask(__name__)
    db_path = db_path or db.DB_PATH
    db.init_db(db_path)
    app.config["MODEL"] = AdClassifier.load()
    app.config["ADMIN_PASSWORD"] = os.environ.get("ADMIN_PASSWORD", "")
    db.load_sample_if_empty(app.config["MODEL"], db_path)

    def model() -> AdClassifier:
        return app.config["MODEL"]

    # -------------------------------------------------------- shared helpers
    def categorize(ads: list[dict]) -> list[dict]:
        """Library category for ads already known (incl. staff corrections), model for new ones."""
        known = db.categories_for([a["id"] for a in ads], db_path)
        with _model_lock:
            fresh = model().predict_many([a for a in ads if a["id"] not in known])
        it = iter(fresh)
        return [{"category": known[a["id"]]} if a["id"] in known else next(it) for a in ads]

    def store(ads: list[dict]):
        with _model_lock:
            db.upsert_ads(ads, model(), db_path)

    from .live import LiveView, Snapshotter
    live_window = LiveView(categorize, category_colors(), store)
    snapshots = Snapshotter(categorize, category_colors(), store, out_dir=db.ROOT / "data" / "snapshots")
    app.config.update(LIVE=live_window, SNAPSHOTS=snapshots)

    def public_ad(a: dict) -> dict:
        out = {k: a.get(k) for k in PUBLIC_AD_FIELDS}
        out["advertiser"] = clean_advertiser(a.get("advertiser", "")) or a.get("domain", "")
        out["landing_url"] = strip_tracking(a.get("landing_url", ""))
        if out["domain"] in NETWORK_DOMAINS:  # search-ad networks: link leads to a results page, not the advertiser
            out["domain"] = out["landing_url"] = ""
        return out

    app.jinja_env.filters["public_ad"] = public_ad

    app.jinja_env.filters["plural"] = lambda n, word: f"{n} {word}" + ("" if n == 1 else "s")
    app.jinja_env.filters["initials"] = lambda name: "".join(w[0] for w in re.findall(r"[A-Za-z0-9]+", name or "")[:2]).upper() or "•"

    @app.context_processor
    def inject():
        return {"colors": category_colors(), "icons": CATEGORY_ICONS, "categories": CATEGORY_NAMES,
                "live_sections": LIVE_SECTIONS}

    # -------------------------------------------------------- staff protection
    def _staff_allowed() -> bool:
        """With ADMIN_PASSWORD set: HTTP Basic login (any user name, that password).
        Without it: only requests made directly on this computer (no proxy in between)."""
        password = app.config["ADMIN_PASSWORD"]
        if password:
            auth = request.authorization
            return bool(auth and auth.password and hmac.compare_digest(auth.password, password))
        return request.remote_addr in ("127.0.0.1", "::1") and "X-Forwarded-For" not in request.headers

    @app.before_request
    def protect_staff_routes():
        if request.path.startswith(("/admin", "/api/admin/")) and not _staff_allowed():
            if app.config["ADMIN_PASSWORD"]:
                return Response("Login required", 401, {"WWW-Authenticate": 'Basic realm="Staff"'})
            abort(404)  # don't reveal that a staff page exists

    # -------------------------------------------------------- customer pages
    def is_network_listing(a: dict) -> bool:
        """Ads a search-ad network runs under its own name (e.g. 'SearchResultOffers') - not a real advertiser."""
        name = re.sub(r"[^a-z0-9]", "", clean_advertiser(a.get("advertiser", "")).lower())
        return a.get("domain") in NETWORK_DOMAINS and (not name or name in a["domain"].replace(".", ""))

    @app.get("/")
    def dashboard():
        date_from = request.args.get("date_from", "").strip()[:10]
        date_to = request.args.get("date_to", "").strip()[:10]
        s = db.stats(db_path, date_from=date_from, date_to=date_to)
        # Brands first: publishers of sponsored stories and search networks are not shown as "top brands".
        s["top_advertisers"] = [a for a in s["top_advertisers"]
                                if not is_network_listing(a) and a["category"] != "Sponsored Stories"][:8]
        # Latest ads: one per category, real brands first, so the row shows the variety of advertising.
        latest, seen = [], set()
        rows = [a for a in db.query_ads(date_from=date_from, date_to=date_to, path=db_path) if a.get("image_url")]
        for a in sorted(rows, key=lambda a: a["category"] == "Sponsored Stories"):
            if a["category"] not in seen:
                seen.add(a["category"]); latest.append(public_ad(a) | {"has_image": True})
            if len(latest) == 8:
                break
        return render_template("dashboard.html", s=s, latest=latest)

    @app.get("/ads")
    def ads():
        f = {k: request.args.get(k, "") for k in ("category", "q")}
        rows = db.query_ads(f["category"], f["q"], path=db_path)
        if not f["category"]:  # "All": brands first, sponsored stories after them
            rows.sort(key=lambda a: a["category"] == "Sponsored Stories")
        counts = {c["category"]: c["n"] for c in db.stats(db_path)["by_category"]}
        pages = max(1, -(-len(rows) // PAGE_SIZE))
        page = min(max(request.args.get("page", 1, type=int), 1), pages)
        shown = [public_ad(a) | {"has_image": bool(a.get("image_url"))}
                 for a in rows[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]]
        return render_template("ads.html", ads=shown, total=len(rows), f=f, page=page, pages=pages, counts=counts,
                               all_total=sum(counts.values()))

    @app.route("/classify", methods=["GET", "POST"])
    def classify():
        form = {k: request.form.get(k, "").strip()[:500] for k in ("title", "description", "advertiser", "url")}
        image_data = request.form.get("image_data", "").strip()
        ocr_text = ""
        raw_bytes = None

        if "image_file" in request.files and request.files["image_file"].filename:
            f = request.files["image_file"]
            raw = f.read(5_000_000)
            if raw:
                import base64
                mime = f.mimetype or "image/jpeg"
                image_data = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
                raw_bytes = raw
        elif image_data and "," in image_data:
            import base64
            try:
                raw_bytes = base64.b64decode(image_data.split(",", 1)[1])
            except Exception:
                raw_bytes = None

        # If ad image was provided and headline is empty, auto-extract with EasyOCR
        if raw_bytes and not form["title"]:
            try:
                from app.ocr import extract_text_from_bytes
                ocr_res = extract_text_from_bytes(raw_bytes)
                if ocr_res.get("headline"):
                    form["title"] = ocr_res["headline"]
                if ocr_res.get("description") and not form["description"]:
                    form["description"] = ocr_res["description"]
                ocr_text = ocr_res.get("full_text", "")
            except Exception:
                pass

        result = None
        ocr_error = None
        if request.method == "POST":
            if form["title"] or form["description"]:
                result = model().predict(form["title"], form["description"], form["advertiser"], _domain(form["url"]))
            elif raw_bytes:
                ocr_error = "Could not detect clear text from this image. Please snap closer to the headline, or switch to 'Type Text'."
        return render_template("classify.html", form=form, result=result, image_data=image_data, ocr_text=ocr_text, ocr_error=ocr_error)

    @app.get("/live")
    def live_page():
        section = request.args.get("section", "/")
        if section not in LIVE_SECTIONS.values():
            section = "/"
        return render_template("live.html", section=section, snap=snapshots.latest(section),
                               fresh=bool(snapshots.fresh(section)))

    @app.get("/live/img/<key>.jpg")
    def live_image(key):
        if not re.fullmatch(r"[a-z0-9-]{1,40}", key):
            abort(404)
        return send_from_directory(snapshots.out_dir, f"{key}.jpg", mimetype="image/jpeg", max_age=60)

    @app.get("/img/<ad_id>")
    def ad_image(ad_id):
        """Ad thumbnails from a local cache instead of hot-linking the ad network."""
        path = _cached_image(ad_id, db_path)
        if path is None:
            abort(404)
        return send_file(path, mimetype="image/jpeg", max_age=7 * 24 * 3600)

    @app.get("/export.csv")
    def export_csv():
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["Category", "Headline", "Description", "Advertiser", "Website", "Link", "First seen", "Last seen"])
        for a in map(public_ad, db.query_ads(path=db_path)):
            w.writerow([a["category"], a["title"], a["description"], a["advertiser"], a["domain"], a["landing_url"],
                        (a["first_seen"] or "")[:10], (a["last_seen"] or "")[:10]])
        name = f"lokmat_ads_report_{datetime.now():%Y%m%d}.csv"
        return Response("﻿" + buf.getvalue(), mimetype="text/csv",  # BOM so Excel shows Marathi correctly
                        headers={"Content-Disposition": f"attachment; filename={name}"})

    # -------------------------------------------------------- customer API
    @app.post("/api/ocr")
    def api_ocr():
        raw = None
        if "image" in request.files and request.files["image"].filename:
            raw = request.files["image"].read(6_000_000)
        elif request.is_json and request.json.get("image_data"):
            data = request.json["image_data"]
            if "," in data:
                data = data.split(",", 1)[1]
            import base64
            try:
                raw = base64.b64decode(data)
            except Exception:
                raw = None
        elif request.form.get("image_data"):
            data = request.form.get("image_data")
            if "," in data:
                data = data.split(",", 1)[1]
            import base64
            try:
                raw = base64.b64decode(data)
            except Exception:
                raw = None
        if not raw:
            return jsonify(error="No image provided"), 400
        try:
            from app.ocr import extract_text_from_bytes
            return jsonify(extract_text_from_bytes(raw))
        except Exception as e:
            return jsonify(error=str(e)), 500

    @app.post("/api/classify")
    def api_classify():
        data = request.get_json(force=True, silent=True) or {}
        if not (data.get("title") or data.get("description")):
            return jsonify(error="Provide at least 'title' or 'description'"), 400
        r = model().predict(str(data.get("title", ""))[:500], str(data.get("description", ""))[:500],
                            str(data.get("advertiser", ""))[:200],
                            data.get("domain") or _domain(str(data.get("url", ""))[:500]))
        return jsonify({k: r[k] for k in ("category", "confidence", "alternatives")})

    @app.get("/api/ads")
    def api_ads():
        rows = db.query_ads(request.args.get("category", ""), request.args.get("q", ""), path=db_path)
        return jsonify([public_ad(a) for a in rows])

    @app.get("/api/stats")
    def api_stats():
        date_from = request.args.get("date_from", "").strip()[:10]
        date_to = request.args.get("date_to", "").strip()[:10]
        s = db.stats(db_path, date_from=date_from, date_to=date_to)
        return jsonify(total_ads=s["total_ads"], advertisers=s["advertisers"], last_updated=s["last_updated"],
                       by_category=[{"category": c["category"], "ads": c["n"]} for c in s["by_category"]])

    @app.post("/api/live")
    def api_live_snapshot():
        section = (request.get_json(force=True, silent=True) or {}).get("section", "/")
        if section not in LIVE_SECTIONS.values():
            return jsonify(error="unknown section"), 400
        return jsonify(state=snapshots.request(section))

    @app.get("/api/live/status")
    def api_live_snapshot_status():
        section = request.args.get("section", "/")
        if section not in LIVE_SECTIONS.values():
            return jsonify(error="unknown section"), 400
        state = ("ready" if snapshots.fresh(section) else "working" if snapshots.busy(section)
                 else "failed" if snapshots.errors.get(snapshots.key(section)) else "idle")
        return jsonify(state=state)

    # -------------------------------------------------------- staff page + API
    @app.get("/admin")
    def admin():
        return render_template("model.html", meta=model().meta, s=db.stats(db_path),
                               review=db.query_ads(review_only=True, path=db_path),
                               refresh_hours=_refresh_hours())

    @app.post("/api/admin/ads/<ad_id>/label")
    def api_label(ad_id):
        category = (request.get_json(force=True, silent=True) or {}).get("category", "")
        if category not in CATEGORY_NAMES:
            return jsonify(error="unknown category"), 400
        ad = db.set_label(ad_id, category, db_path)
        if ad is None:
            return jsonify(error="ad not found"), 404
        model().remember(ad["advertiser"], ad["domain"], category)  # effective immediately, before retraining
        return jsonify(ok=True, category=category)

    @app.post("/api/admin/retrain")
    def api_retrain():
        with _model_lock:
            new_model = AdClassifier.train(extra_rows=db.feedback_rows(db_path))
            app.config["MODEL"] = new_model
            n = db.reclassify_all(new_model, db_path)
        return jsonify(ok=True, reclassified=n, accuracy=new_model.meta["evaluation"]["accuracy"],
                       n_samples=new_model.meta["n_samples"])

    @app.post("/api/admin/refresh")
    def api_refresh():
        return jsonify(ok=True, started=_start_refresh(app, db_path))

    @app.get("/api/admin/refresh/status")
    def api_refresh_status():
        return jsonify(running=_refresh_state["running"], result=_refresh_state["result"])

    @app.post("/api/admin/live-window")
    def api_live_window():
        section = (request.get_json(force=True, silent=True) or {}).get("section", "/")
        if section not in LIVE_SECTIONS.values():
            return jsonify(error="unknown section"), 400
        live_window.open(section)  # no-op if a window is already open
        return jsonify(ok=True)

    @app.get("/api/admin/live-window/status")
    def api_live_window_status():
        return jsonify(running=live_window.running, failed=bool(live_window.error))

    # -------------------------------------------------------- errors
    @app.errorhandler(404)
    def not_found(_):
        return render_template("error.html", code=404, message="We couldn’t find that page."), 404

    @app.errorhandler(500)
    def server_error(_):
        return render_template("error.html", code=500, message="Something went wrong. Please try again."), 500

    if auto_refresh if auto_refresh is not None else _refresh_hours() > 0:
        threading.Thread(target=_auto_refresh_loop, args=(app, db_path), daemon=True).start()
    return app


# ------------------------------------------------------------ ad refresh
def _refresh_hours() -> float:
    try:
        return float(os.environ.get("REFRESH_HOURS", "6"))
    except ValueError:
        return 6.0


def _start_refresh(app: Flask, db_path) -> bool:
    with _refresh_lock:
        if _refresh_state["running"]:
            return False
        _refresh_state.update(running=True, log=[], result=None, run_id=db.start_run(db_path))
    threading.Thread(target=_run_refresh, args=(app, db_path), daemon=True).start()
    return True


def _auto_refresh_loop(app: Flask, db_path):
    """Keep the library current without anyone pressing a button."""
    time.sleep(90)  # let the server finish starting
    while True:
        hours = _refresh_hours()
        if hours <= 0:
            return
        last = db.last_successful_run(db_path)
        if last is None or datetime.now() - last >= timedelta(hours=hours):
            _start_refresh(app, db_path)
        time.sleep(15 * 60)


def _run_refresh(app: Flask, db_path):
    """Collect the latest ads in the background. Details go to the server console and the
    scrape_runs table; customers only ever see the resulting ads and the 'Last updated' date."""
    from .scraper import scrape

    log = _refresh_state["log"]

    def note(line: str):
        log.append(line)
        print(line, flush=True)

    status, found, new = "failed", 0, 0
    try:
        ads = scrape(log=note)
        found = len(ads)
        with _model_lock:
            new = db.upsert_ads(ads, app.config["MODEL"], db_path)
        note(f"Stored: {new} new ads, {found - new} already known.")
        status = "success" if found else "failed"
    except Exception as exc:
        note(f"ERROR: {type(exc).__name__}: {exc}")
        if "Google Chrome is needed" in str(exc):
            note("Fix: install Google Chrome from https://www.google.com/chrome/")
    finally:
        db.finish_run(_refresh_state["run_id"], status, found, new, "\n".join(log), db_path)
        _refresh_state.update(running=False, result={"status": status, "new": new})


# ------------------------------------------------------------ helpers
def _cached_image(ad_id: str, db_path):
    import urllib.request

    if not re.fullmatch(r"[0-9a-f]{16}", ad_id):
        return None
    cache = db.ROOT / "data" / "images"
    path = cache / f"{ad_id}.jpg"
    if path.exists():
        return path
    with db.connect(db_path) as c:
        row = c.execute("SELECT image_url FROM ads WHERE id=?", (ad_id,)).fetchone()
    if not row or not (row["image_url"] or "").startswith("http"):
        return None
    try:
        req = urllib.request.Request(row["image_url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = r.read(3_000_000)
    except Exception:
        return None
    cache.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _domain(url: str) -> str:
    from .scraper import _domain as d
    if url and "://" not in url:
        url = "http://" + url
    return d(url) if url else ""
