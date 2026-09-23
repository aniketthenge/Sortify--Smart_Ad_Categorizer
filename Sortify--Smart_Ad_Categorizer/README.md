# Lokmat Times Ad Categorizer

A web application that **collects the live advertisements shown on [lokmattimes.com](https://www.lokmattimes.com)** and
**automatically sorts them into 21 business categories** (Finance, Insurance, Real Estate, Matrimony, Clickbait, and so on).
It includes a dashboard, a searchable ad browser, a "paste any ad" classifier, an HTTP API, and a feedback loop: you correct
wrong labels and the model retrains on them.

It ships with **219 real ads scraped from Lokmat Times on 23 Sep 2026**, so the dashboard has data from the first launch.
You can pull fresh ads at any time with one click.

---

## 1. Quick start (Windows)

**Requirements:** Python 3.10 or newer ([python.org](https://www.python.org/downloads/), tick *"Add Python to PATH"* when installing) and an internet connection.

1. Unzip `ad_categorizer.zip` anywhere.
2. Double-click **`setup.bat`** (one time only, about 2–4 minutes). It:
   - creates a private virtual environment in `.venv\`
   - installs Flask, scikit-learn and Playwright
   - downloads a headless Chromium (~115 MB) for the scraper
   - trains the model
3. Double-click **`run.bat`**. Your browser opens at **http://127.0.0.1:5000**.

Stop the server with `Ctrl+C` in the console window.

### Manual setup (Windows / macOS / Linux)

```bash
cd ad_categorizer
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
python cli.py train
python run.py            # -> http://127.0.0.1:5000
```

---

## 2. Using the app

| Page | What it does |
|---|---|
| **Dashboard** | KPIs, ads per category, top advertisers, ad networks, scrape history. **"Scrape live ads now"** crawls ~10 Lokmat Times sections (2–4 min) and streams progress. **"Export CSV"** downloads everything (opens correctly in Excel, including Marathi). |
| **Ads** | Every unique ad with thumbnail, advertiser, landing page, times seen, category and confidence. Filter by category, network, search text, or *Needs review* (confidence under 35%). **Change the drop-down to correct a category**; it is saved immediately. |
| **Classify** | Paste any headline, description, advertiser or URL (English / Marathi / Hindi) to get the category, confidence, top-3 alternatives and the reason (matched keywords + most influential model terms). |
| **Model** | Cross-validated accuracy, per-category precision/recall/F1, training-data balance. **"Retrain with my corrections"** retrains in about 15 s using your labels and re-classifies all stored ads. |

### Command line

```bash
python cli.py scrape                         # all default sections
python cli.py scrape --sections / /business/ # only some sections
python cli.py scrape --show-browser          # watch Chromium work
python cli.py train                          # retrain (includes UI corrections)
python cli.py classify "3 BHK flats in Wakad" --advertiser "Kolte Patil" --domain koltepatil.com
python cli.py stats
python cli.py export ads.csv
```

### HTTP API

```bash
curl -X POST http://127.0.0.1:5000/api/classify -H "Content-Type: application/json" \
     -d '{"title":"Term insurance on easy EMI","advertiser":"Policybazaar","url":"https://policybazaar.com"}'
```

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/classify` | `{title, description?, advertiser?, url? or domain?}` → prediction JSON |
| GET | `/api/ads?category=&q=` | stored ads |
| GET | `/api/stats` | dashboard numbers |
| POST | `/api/ads/<id>/label` | `{category}`: human correction |
| POST | `/api/retrain` | retrain + re-classify |
| POST | `/api/scrape` | start background scrape (`{sections: [...]}` optional) |
| GET | `/api/scrape/status` | progress log |
| GET | `/export.csv` | CSV download |

---

## 3. How it works

```
lokmattimes.com ──► Playwright (headless Chromium) ──► Taboola sponsored cards + Google ad iframes
                                                              │  dedupe (title+advertiser+domain hash)
                                                              ▼
                     TF-IDF word 1-2 grams + char 2-5 grams ──► Logistic Regression (75%)
                     Keyword lexicon per category ────────────► rule score          (25%)
                                                              │
                                                              ▼
                     SQLite (data/ads.db) ──► Flask UI / API / CSV ──► human corrections ──► retrain
```

**Why a headless browser?** Lokmat Times injects its ads with JavaScript (Taboola native widgets, Google Ad Manager).
A plain HTTP request returns the page *without* any ads. The scraper opens each section, scrolls to trigger lazy loading,
then reads the rendered widgets. Taboola's organic article recommendations (links back to lokmattimes.com) are discarded;
only cards marked as sponsored (`syndicatedItem`) are kept.

**Why hybrid ML + rules?** Ads are short (often a 6–10 word headline), so a pure statistical model trained on a few hundred
examples guesses poorly on advertisers it has never seen. Adding a small curated keyword lexicon raises held-out accuracy
from about 49% to about 87%, and it also explains each decision in plain terms.

**Categories (21):** Finance & Investment · Insurance · Health & Wellness · Matrimony & Dating · Gaming · Education & Courses ·
Real Estate · Home & Living · Fashion & Shopping · Technology & Electronics · Automotive · Travel & Tourism · Charity & NGO ·
Astrology & Spirituality · Industrial & B2B · Food & Beverage · Entertainment & Media · Professional Services ·
Jobs & Careers · Government & Public Notices · Sponsored Content / Clickbait.

### Training data (`data/training_data.csv`)
- **193 real Lokmat Times ads**, hand-labelled.
- **159 curated examples** for categories that were rare in the live sample but common in Indian news advertising
  (jobs, government notices, education, automotive, real estate…), including Marathi text.

### Evaluation
5-fold **GroupKFold by advertiser domain**: an advertiser is never in both the training and test folds. This measures
performance on *new advertisers* rather than memorised ones.

| Metric | Value |
|---|---|
| Accuracy (hybrid) | **86.9%** |
| Macro F1 | **0.86** |
| Accuracy (ML only, no rules) | 49.1% |

Weakest categories: Automotive and Entertainment (F1 ≈ 0.7). They have the fewest examples.

---

## 4. Project layout

```
ad_categorizer/
├── run.py                 # start web server
├── cli.py                 # command-line interface
├── setup.bat / run.bat    # Windows one-click setup & launch
├── requirements.txt
├── app/
│   ├── scraper.py         # Playwright scraper for lokmattimes.com
│   ├── classifier.py      # categories, keyword rules, ML pipeline, evaluation
│   ├── db.py              # SQLite storage
│   ├── web.py             # Flask routes + JSON API + background scrape job
│   ├── templates/         # dashboard, ads, classify, model pages
│   └── static/            # CSS + JS (no CDN, works offline)
├── data/
│   ├── training_data.csv  # labelled training set
│   ├── sample_ads.json    # 219 ads scraped 23 Sep 2026 (loaded on first run)
│   └── ads.db             # created at runtime
├── models/                # trained model (created at runtime)
└── tests/test_app.py      # 18 pytest tests
```

Run the tests with `python -m pytest`.

---

## 5. Limitations and notes

- **Which ads you get depends on the ad servers.** Taboola and Google choose creatives per visitor, location and time.
  From an Indian IP you will see Indian advertisers. Running from elsewhere shows different ads.
- **Google display ads are best-effort.** Most arrive as images inside cross-origin frames, and many do not fill in a headless
  browser, so almost all captured ads are Taboola native ads (which carry text and are the bulk of Lokmat's inventory).
  Image-only banners would need OCR, which is a possible extension.
- **If Lokmat Times or Taboola changes its HTML**, update the selectors in `TABOOLA_JS` in `app/scraper.py`. There is a
  text-based fallback for the headline and advertiser.
- **The keyword lexicon was written with knowledge of the sample ads**, so the 86.9% figure is somewhat optimistic for
  unfamiliar ad types. Use the *Needs review* filter and the correction → retrain loop to improve it on your own data.
- Scraping is light (about 10 page loads per run, sequential) and only reads public pages. Check Lokmat Times' terms of use
  before running it on a schedule or at scale.

### Troubleshooting

| Problem | Fix |
|---|---|
| `python` not recognised | Reinstall Python with "Add to PATH" ticked, or use `py` instead of `python`. |
| Scrape log says *Executable doesn't exist* | `.venv\Scripts\python -m playwright install chromium` |
| Port 5000 busy | `set PORT=8000` then `run.bat` |
| Want a clean start | Delete `data\ads.db`; the sample ads reload on next launch. |
| Model error after upgrading packages | Delete `models\ad_classifier.joblib`; it rebuilds automatically. |
