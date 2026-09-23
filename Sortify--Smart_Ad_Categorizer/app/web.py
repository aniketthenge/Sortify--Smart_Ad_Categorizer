"""Flask web application: dashboard, ad browser, live classifier, model page."""
from __future__ import annotations

import csv
import io
import threading
from datetime import datetime

from flask import Flask, Response, jsonify, redirect, render_template, request, url_for

from . import db
from .classifier import CATEGORY_NAMES, AdClassifier, category_colors

_model_lock = threading.Lock()
_scrape_state = {"running": False, "log": [], "run_id": None, "result": None}
_scrape_lock = threading.Lock()


def create_app(db_path=None) -> Flask:
    app = Flask(__name__)
    db_path = db_path or db.DB_PATH
    db.init_db(db_path)
    app.config["MODEL"] = AdClassifier.load()
    db.load_sample_if_empty(app.config["MODEL"], db_path)

    def model() -> AdClassifier:
        return app.config["MODEL"]

    @app.context_processor
    def inject():
        return {"colors": category_colors(), "categories": CATEGORY_NAMES}

    # ------------------------------------------------------------ pages
    @app.get("/")
    def dashboard():
        return render_template("dashboard.html", s=db.stats(db_path), meta=model().meta)

    @app.get("/ads")
    def ads():
        f = {k: request.args.get(k, "") for k in ("category", "q", "network")}
        review = request.args.get("review") == "1"
        rows = db.query_ads(f["category"], f["q"], review, f["network"], path=db_path)
        return render_template("ads.html", ads=rows, f=f, review=review)

    @app.route("/classify", methods=["GET", "POST"])
    def classify():
        form = {k: request.form.get(k, "").strip() for k in ("title", "description", "advertiser", "url")}
        result = None
        if request.method == "POST" and (form["title"] or form["description"]):
            domain = _domain(form["url"])
            result = model().predict(form["title"], form["description"], form["advertiser"], domain)
        return render_template("classify.html", form=form, result=result)

    @app.get("/model")
    def model_page():
        return render_template("model.html", meta=model().meta, s=db.stats(db_path))

    @app.get("/export.csv")
    def export_csv():
        rows = db.query_ads(path=db_path)
        buf = io.StringIO()
        cols = ["id", "category", "confidence", "method", "manual_label", "title", "description", "advertiser",
                "domain", "landing_url", "network", "times_seen", "first_seen", "last_seen", "source_page"]
        w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        name = f"lokmat_ads_{datetime.now():%Y%m%d_%H%M}.csv"
        return Response("﻿" + buf.getvalue(), mimetype="text/csv",  # BOM so Excel shows Marathi correctly
                        headers={"Content-Disposition": f"attachment; filename={name}"})

    # ------------------------------------------------------------ API
    @app.post("/api/classify")
    def api_classify():
        data = request.get_json(force=True, silent=True) or {}
        if not (data.get("title") or data.get("description")):
            return jsonify(error="Provide at least 'title' or 'description'"), 400
        domain = data.get("domain") or _domain(data.get("url", ""))
        return jsonify(model().predict(data.get("title", ""), data.get("description", ""),
                                       data.get("advertiser", ""), domain))

    @app.post("/api/ads/<ad_id>/label")
    def api_label(ad_id):
        category = (request.get_json(force=True, silent=True) or request.form).get("category", "")
        if category not in CATEGORY_NAMES:
            return jsonify(error="unknown category"), 400
        if not db.set_label(ad_id, category, db_path):
            return jsonify(error="ad not found"), 404
        return jsonify(ok=True, category=category)

    @app.post("/api/retrain")
    def api_retrain():
        with _model_lock:
            new_model = AdClassifier.train(extra_rows=db.feedback_rows(db_path))
            app.config["MODEL"] = new_model
            n = db.reclassify_all(new_model, db_path)
        return jsonify(ok=True, reclassified=n, meta=new_model.meta)

    @app.post("/api/scrape")
    def api_scrape():
        with _scrape_lock:
            if _scrape_state["running"]:
                return jsonify(error="A scrape is already running"), 409
            _scrape_state.update(running=True, log=[], result=None, run_id=db.start_run(db_path))
        sections = request.get_json(force=True, silent=True) or {}
        threading.Thread(target=_run_scrape, args=(app, db_path, sections.get("sections")), daemon=True).start()
        return jsonify(ok=True, run_id=_scrape_state["run_id"])

    @app.get("/api/scrape/status")
    def api_scrape_status():
        return jsonify(running=_scrape_state["running"], log=_scrape_state["log"][-200:],
                       result=_scrape_state["result"])

    @app.get("/api/stats")
    def api_stats():
        return jsonify(db.stats(db_path))

    @app.get("/api/ads")
    def api_ads():
        return jsonify(db.query_ads(request.args.get("category", ""), request.args.get("q", ""), path=db_path))

    @app.get("/scrape")
    def scrape_redirect():
        return redirect(url_for("dashboard"))

    return app


def _run_scrape(app: Flask, db_path, sections):
    from .scraper import scrape

    log = _scrape_state["log"]
    status, found, new = "failed", 0, 0
    try:
        ads = scrape(sections, log=log.append)
        found = len(ads)
        with _model_lock:
            new = db.upsert_ads(ads, app.config["MODEL"], db_path)
        log.append(f"Stored: {new} new ads, {found - new} already known.")
        status = "success"
    except Exception as exc:  # surface errors (e.g. Chromium not installed) in the UI
        log.append(f"ERROR: {type(exc).__name__}: {exc}")
        if "Executable doesn't exist" in str(exc):
            log.append("Fix: run  .venv\\Scripts\\python -m playwright install chromium")
    finally:
        db.finish_run(_scrape_state["run_id"], status, found, new, "\n".join(log), db_path)
        _scrape_state.update(running=False, result={"status": status, "found": found, "new": new})


def _domain(url: str) -> str:
    from .scraper import _domain as d
    if url and "://" not in url:
        url = "http://" + url
    return d(url) if url else ""
