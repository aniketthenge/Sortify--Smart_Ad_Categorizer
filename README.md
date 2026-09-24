# Lokmat Times Ad Categorizer

A web application that **collects the advertisements running on [lokmattimes.com](https://www.lokmattimes.com)** and
**automatically sorts them into 21 business categories** (Finance, Insurance, Real Estate, Matrimony, Sponsored Stories, and so on).

- **Customer site:** Overview · Ad Library · Live Site · Categorize an Ad
- **Staff page:** `/admin` (password-protected, never linked from the customer site). Shows accuracy, lets staff correct
  low-confidence ads, retrain the model, and collect ads on demand.

It ships with **411 real ads collected from Lokmat Times on 23–24 Sep 2026**, and refreshes itself every 6 hours.

**Accuracy today:** 96.1% on advertisers it has never seen, 98.1% on new ads from advertisers it knows,
93.7% on a separate Marathi/Hindi test set. Details: [Accuracy](#accuracy).

**Start here:**
[What has been done](#what-has-been-done-change-log) ·
[Check it yourself, step by step](#check-it-yourself-step-by-step) ·
[How the categorization works, in plain words](#how-the-categorization-works-in-plain-words) ·
[Go-live checklist in plain words](#2-going-live-for-customers) ·
[Quick start](#1-quick-start-windows-on-your-own-pc)

---

## What has been done (change log)

Newest first. Each entry says what changed and why.

**Round 7: Google Chrome only (24 Sep 2026)**
- **The app now uses Google Chrome strictly** for everything that drives a browser: ad collection, the Live Site capture
  and the staff demo window. No Edge, and no separately downloaded browser. Chrome is signed by Google, so Windows Smart
  App Control allows it.
- **Starting the app opens the site in Chrome** (your default browser is used only if Chrome isn't installed).
- **Setup no longer downloads a 115 MB browser**; `setup.bat` checks that Chrome is installed and tells you if it isn't.
  If Chrome is missing, Live Site / ad collection show a clear "install Google Chrome" message in the console.
- **Checked in Chrome:** ad collection (78 ads from 2 sections), Live Site capture (74 ads on the Pune page), the staff
  "Open in a window" demo (opened `chrome.exe`, labelled ads, detected closing), and the layout inspector on every page at
  5 screen widths: **0 problems**.
- **Checked in Edge as well** (for customers who use Edge to *view* the site): same inspection, **0 problems**, and the
  pages look identical.

**Round 6: new, attractive customer interface (24 Sep 2026)**
- **Redesigned every customer page:** picture cards for ads, category tiles with icons on the Overview, "Latest ads" and
  "Top brands" sections, category chips with counts in the Ad Library, page tabs on the Live Site, and a clear result
  card on Categorize an Ad.
- **One colour + icon per category** (e.g. 🚗 Automotive, 🏠 Real Estate), used everywhere. All 21 colours were checked
  against the accessibility contrast standard (WCAG 4.5:1), so labels are easy to read.
- **Less noise:** removed the "Categories: 19" number; the raw "how sure" percentage is replaced by a plain
  *Strong / Good / Possible match* label; "could also be" only shows real alternatives; no ad-network names anywhere.
- **Real brands first:** the Ad Library ("All") and "Latest ads" show brand ads before sponsored stories, and "Top brands"
  lists real advertisers only. Sponsored stories are still there under their own category.
- **Wording fixes:** "1 ad" / "74 ads" (was "1 ads"), clearer Live Site hint.
- **Phones:** compact number tiles, category names on their own line, 4 latest ads instead of 8.
- **Checked automatically:** a layout inspector opened 10 page states at 5 screen widths (1440, 1280, 1024, 820 and
  390 px) and checked for overlapping text, text spilling out of boxes, sideways scrolling and broken images.
  Result: **0 problems**.
- **Reliability fix:** Windows Smart App Control started blocking the browser the app uses for Live Site and ad
  collection. (Round 7 replaced this with Google Chrome only.)

**Round 5: higher accuracy, better Marathi/Hindi (24 Sep 2026)**
- **Stronger language model:** switched to Google's **EmbeddingGemma-300m** (multilingual). Five candidate models were
  compared on the same test; it was the most accurate and 3× faster than the runner-up.
  On the same 684 examples: **89.6% → 96.1%** on unseen advertisers, **94.9% → 98.1%** on known advertisers.
- **Marathi/Hindi:** built a separate test set of 63 Marathi/Hindi ads that are never used for training
  (`data/eval_marathi_hindi.csv`). Score: **79.4% → 93.7%**.
- **Fixed a Marathi keyword bug:** short words matched inside other words; e.g. "वर" (groom) inside "कपड्यांवर"
  ("on clothes") pushed a clothing sale towards Matrimony. Keywords now have to start a word. Removed "सूट" from
  Fashion (in Marathi it usually just means "discount").
- **More training data:** 46 more real Lokmat ads (hand-labelled) + 81 new Marathi/Hindi examples → 684 examples.
- **Safe fallback:** if EmbeddingGemma can't be downloaded, the app automatically uses the previous model (MiniLM),
  then word-matching only. It never fails to start because of this.
- **Cost:** first-time download is now ~1.2 GB instead of ~0.2 GB, and the running app uses ~1.8 GB of memory
  (was ~0.5 GB). Categorising one ad still takes ~0.03 s.
- Tests now build the sample library once, so the full test run takes ~1 minute instead of ~10.
- Fixed: a model-saving path that tools could not redirect (caused my measurement run to overwrite the app's model; caught and retrained).

**Round 4: made safe for customers (24 Sep 2026)**
- **Staff page locked.** `/admin` now needs a password (`ADMIN_PASSWORD`). Without one, it works only on the computer
  running the app and shows "Page not found" to everyone else. *Why:* before, any visitor could change categories or retrain the model.
- **Nothing internal reaches customers.** Customers' browsers no longer receive the staff JavaScript, ad-network names,
  which Lokmat page an ad came from, model confidence/method, or collection history. *Why:* a customer can see all of
  that with "View page source" or by opening the data links, even if it isn't on screen.
- **"Refresh ads" button removed from customers.** Ads now refresh by themselves every 6 hours; customers see "Last updated".
  Staff can still collect on demand from `/admin`. *Why:* any visitor could start a 4-minute job on the server.
- **New Live Site page** that works from any computer or phone: the real Lokmat page as it looks now, every ad labelled,
  with a clickable list. *Why:* the earlier "View live site" opened a window on the server, so a remote customer saw nothing.
  The window version is kept on the staff page for office demos.
- **Production server:** runs on Waitress; `HOST` setting lets other computers connect.
- **Ad Library** shows 48 ads per page; search-network ads (srchly.co etc.) shown without their tracking link;
  networks no longer listed as "top advertisers".
- Friendly "Page not found" / error pages.
- Automated test that fails if any customer-facing page, script, data feed, CSV or error page mentions internals.

**Round 3: live site with ads labelled in place (24 Sep 2026)**
- "View live site" opened lokmattimes.com in a browser window with each ad outlined and badged by category, plus a
  "Ads on this page" panel. (Now the staff demo; customers use the Live Site page from round 4.)

**Round 2: accuracy, white theme, clean customer UI (24 Sep 2026)**
- **Accuracy 80.4% → 92.3%** on advertisers the model has never seen (96.9% for new ads from known advertisers).
  How: 143 more real ads hand-labelled (training set 352 → 557), a multilingual language model (understands meaning,
  incl. Marathi/Hindi), broader keyword lists, and advertiser memory.
- **White theme always** (it had followed Windows dark mode).
- **Removed from the customer UI:** footer, "How it works", collection log, ad-network names, scrape history, API box,
  confidence/method labels, "seen N×", "needs review", impressions.
- "Sponsored Content / Clickbait" renamed **"Sponsored Stories"**.
- Staff tools moved to `/admin`.
- Fixed a Windows Smart App Control block (rewrote one scikit-learn component in NumPy; accuracy unchanged).

**Round 1: first version (23 Sep 2026)**
- Collector for lokmattimes.com ads, 21 categories, dashboard, ad browser, "categorize an ad" page, CSV export, SQLite
  storage, command line, `setup.bat` / `run.bat`.

---

## Check it yourself, step by step

Takes about 15 minutes. Each step says what you should see. If something differs, note the step number.

**A. Start it**
1. Double-click `run.bat`. A black console window opens and then **Google Chrome** opens at `http://127.0.0.1:5000`.
   *(First time only: 5–10 minutes of setup first, mostly downloads.)*

**B. Customer pages**. Check what's there *and what isn't*.
2. **Overview.** You should see 3 number tiles (Ads, Advertisers, Last updated), a grid of **category tiles** with an
   icon, count and "% of all ads", a row of **Latest ads** (picture cards, one per category), **Top brands**, and two
   buttons: **View live site**, **Download report**.
   You should **not** see: a Refresh button, any mention of scraping/Taboola/model/confidence, or a footer.
3. Turn on Windows dark mode (Settings → Personalization → Colors → Dark) and reload. The site should **stay white**.
4. Click a category tile (e.g. *Health & Wellness*). The Ad Library opens filtered to that category.
5. **Ad Library.** Search `insurance`, then click a category chip under the search box, then **Clear filters**.
   The chip row scrolls sideways (it fades at the right edge). Go to the bottom and click **Next ›**: "Page 2 of …".
   Each ad is a picture card with its category label, headline, advertiser and **Visit ↗**.
6. **Live Site.** Click the *Chhatrapati Sambhajinagar* tab. A message says it takes about half a minute; then the Lokmat page appears with each ad
   outlined and labelled, and a list on the left ("NN ads on this page"). Click any entry: the page scrolls to that
   ad and highlights it. Entries with **×2** jump to each place that ad appears when clicked again.
7. **Categorize an Ad.** Click the example chips, then type your own, e.g. `नई SUV पर भारी छूट` → *Automotive*;
   `2 BHK flats in Baner` → *Real Estate*. The result shows the category's icon and name and a *Strong / Good /
   Possible match* label.
8. **Download report** on the Overview. The CSV opens in Excel with columns Category, Headline, Description,
   Advertiser, Website, Link, First seen, Last seen. Marathi text should display correctly.
9. **Page source.** On any page press `Ctrl+U` and search (`Ctrl+F`) for `taboola`, `scrap`, `admin`, `retrain`.
   There should be no matches, with two known exceptions that come from the ads themselves, not from the app:
   one advertiser's own web address contains "taboola" (TradeWise, Ad Library pages 2/4/6), and one ad headline
   contains "Administration" (Ad Library page 2).
10. Open `http://127.0.0.1:5000/api/ads`. The data has only: id, title, description, advertiser, domain, landing_url,
    category, first_seen, last_seen.
11. Open `http://127.0.0.1:5000/no-such-page`. You should get a friendly "Page not found" page.

**C. Staff page**
12. Open `http://127.0.0.1:5000/admin` on the same PC. You should see accuracy (**96.1%**), per-category accuracy,
    low-confidence ads with a drop-down, **Retrain**, **Collect latest ads now**, and **Open in a window**.
13. Change a low-confidence ad's category in the drop-down ("Saved" appears), then press **Retrain** (about a minute).
14. Press **Open in a window**. A browser window opens on Lokmat Times; scroll and watch ads get labelled. Close it.

**D. Staff page is hidden from other people** (needs a phone or second PC on the same Wi-Fi)
15. Close the console window. Open a Command Prompt in the project folder and run:
    ```bat
    set HOST=0.0.0.0
    .venv\Scripts\python run.py
    ```
    If Windows Firewall asks, allow *Private networks*. Find your PC's address with `ipconfig` ("IPv4 Address", e.g. 192.168.1.20).
16. On the phone open `http://192.168.1.20:5000`. The customer site works.
    Open `http://192.168.1.20:5000/admin`. You should get **"Page not found"**.
17. Stop it (`Ctrl+C`), run `set ADMIN_PASSWORD=test123`, start again. On the phone `/admin` now asks for a login:
    any user name + `test123` opens it; a wrong password doesn't.

**E. Automated checks**
18. In the project folder run `.venv\Scripts\python -m pytest`. Expected: **38 passed** (takes ~1 minute). These
    include the "customers see no internals", "staff page protected" and Marathi/Hindi accuracy (≥ 90%) checks.

---

## 1. Quick start

**Requirements:** Python 3.10 or newer ([python.org](https://www.python.org/downloads/), tick *"Add Python to PATH"*),
**Google Chrome** ([google.com/chrome](https://www.google.com/chrome/)) and an internet connection for the first run.

```bash
# 1. Create and activate virtual environment
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (downloads multilingual embeddings on first run)
python cli.py train

# 4. Start the web application
python run.py
```

Your browser opens at **http://127.0.0.1:5000**. The staff page is at **http://127.0.0.1:5000/admin**.
Stop the server with `Ctrl+C` in the console window.

---

## 2. Going live for customers

### In plain words: three things to do before customers use it

**1. Run it on a server, with two settings.**
Right now the app runs on *your PC*, so only you can open it. For customers to open it from their own computers and
phones, it must run on a computer that is **always on and reachable over the internet**: a company server or a rented
cloud machine (AWS, Azure, Google Cloud…). Your IT team usually provides this. On that machine, set two settings
before starting the app:
- `HOST=0.0.0.0` means "let visitors from other computers in". Without it, the app only answers the machine it runs on.
- `ADMIN_PASSWORD=<a long password>` is the password staff type to open the staff page. Customers don't have it.

**2. Add HTTPS (the padlock 🔒 in the address bar).**
Without HTTPS, everything between the browser and the app, including the staff password, travels in readable form,
and someone on the same network could read it. IT normally sets this up together with a web address such as
`ads.yourcompany.com`, using a "reverse proxy" (IIS, nginx or Caddy) that sits in front of the app. It's standard, one-time
setup; the app itself doesn't change.

**3. Check you're allowed to use Lokmat's pages this way.**
The app automatically visits lokmattimes.com every 6 hours and copies information about the ads on its pages. For
internal use that's low-risk. Offering it to customers as a service built on another company's website may need
Lokmat's permission. Check the Terms of Use on their site or ask your legal team. If Lokmat itself is the customer, get
their OK in writing.

**Server size:** the app needs about **2 GB of memory** while running (the language model), so pick a
machine with at least 4 GB RAM and ~3 GB free disk.

**Nice to have:** make it restart automatically after a reboot (IT: "run it as a service"), and back up the file
`data/ads.db` regularly (it holds the ad library and staff corrections).

### Technical details (for IT)

`run.py` serves the app with **Waitress** (a production WSGI server). Configure it with environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `HOST` | `127.0.0.1` | `0.0.0.0` to accept connections from other computers |
| `PORT` | `5000` | port to listen on |
| `ADMIN_PASSWORD` | *(none)* | **required to use `/admin` from other computers.** Staff log in with any user name + this password. Without it, `/admin` only works on the server itself and returns "not found" to everyone else. |
| `REFRESH_HOURS` | `6` | how often ads are collected automatically (`0` = off) |

Windows example:

```bat
set HOST=0.0.0.0
set ADMIN_PASSWORD=choose-a-long-password
.venv\Scripts\python run.py
```

Recommended for an internet-facing deployment:
- Put it behind **HTTPS** (IIS, nginx or Caddy as a reverse proxy, or your company's load balancer). Basic-auth passwords
  must not travel over plain HTTP. Behind a proxy the "on this computer only" rule for `/admin` is switched off
  automatically, so **set `ADMIN_PASSWORD`**.
- Run it as a service (NSSM or Task Scheduler on Windows, systemd on Linux) so it restarts after reboots.
- Back up `data/ads.db` (the ad library and staff corrections).

---

## 3. Pages

| Page | Who | What it does |
|---|---|---|
| **Overview** `/` | customers | Total ads, advertisers, categories, last update; ads per category; top advertisers; **Download report** (CSV that opens correctly in Excel, including Marathi). |
| **Ad Library** `/ads` | customers | Every ad with image, advertiser and category; search, filter by category, 48 per page. |
| **Live Site** `/live` | customers | The real Lokmat Times page (Home, Chhatrapati Sambhajinagar, Pune, Business…) as it looks right now, with **every ad outlined and labelled by category**, plus a list of the ads on that page. Click an ad in the list to jump to it on the page (repeats are counted, e.g. ×2). The page is captured on the server in about 25–30 s and reused for 10 minutes, so it works from any customer's computer and phone. |
| **Categorize an Ad** `/classify` | customers | Enter a headline (English / Marathi / Hindi) to get its category, match %, and other possible categories. |
| **Staff** `/admin` | staff | Accuracy per category; low-confidence ads with a drop-down to correct them (applies to that advertiser immediately); **Retrain**; **Collect latest ads now** with run history; **Open in a window**: the live site in a real browser window on the server PC, labelled as you scroll (for office demos). |

**What customers never see:** how ads are collected, the ad networks involved, model details, confidence flags,
staff functions, or error details. This covers the pages, the JavaScript and CSS they download, the JSON API, the CSV and
the error pages. `tests/test_app.py` checks all of them on every test run. Errors are written to the server console.

### Command line

```bash
python cli.py scrape                         # collect ads from all default sections
python cli.py scrape --sections / /business/ # only some sections
python cli.py train                          # retrain (includes staff corrections)
python cli.py classify "3 BHK flats in Wakad" --advertiser "Kolte Patil" --domain koltepatil.com
python cli.py stats
python cli.py export ads.csv
```

### HTTP API

Public (customer-safe fields only):

| Method | Endpoint | Returns |
|---|---|---|
| POST | `/api/classify` `{title, description?, advertiser?, url?}` | category, confidence, alternatives |
| GET | `/api/ads?category=&q=` | ads: title, description, advertiser, website, link, category, dates |
| GET | `/api/stats` | totals and ads per category |
| POST | `/api/live` `{section}` · GET `/api/live/status?section=` | Live Site capture |

Staff (protected like `/admin`): `POST /api/admin/ads/<id>/label`, `POST /api/admin/retrain`,
`POST /api/admin/refresh`, `POST /api/admin/live-window` (+ `/status` for each background job).

---

## How the categorization works, in plain words

Think of three experts looking at the same ad, plus a notebook:

1. **The "meaning" expert** (a language model from Google called EmbeddingGemma). It reads the headline and understands
   what it's *about*, in English, Marathi or Hindi. It knows "SUV", "car", "गाडी" and "test drive" are all about cars, even
   if that exact wording was never seen before. We showed it 684 example ads with the correct category (382 real Lokmat
   ads we labelled by hand, plus examples we wrote), so it learned what each category "sounds like".
2. **The "exact words" expert.** It looks for specific words and word pieces that were typical of each category in those
   684 examples, e.g. "matrimony" hidden inside the web address *elitematrimony.com*.
3. **The keyword list.** A hand-made list of tell-tale words per category, like *BHK, flats, possession* → Real Estate,
   or *donate, 80G* → Charity. It only speaks up when one of its words appears.

The first two are combined (3 parts meaning, 1 part exact words), and the keyword list then adds its vote.
The category with the highest combined score wins.

4. **The notebook (advertiser memory).** Advertisers almost never change category. If we've seen this advertiser before,
   or a staff member corrected one of its ads on `/admin`, that counts as a strong extra vote.

**Example:** *"नई SUV पर भारी छूट, आज ही टेस्ट ड्राइव बुक करें"* ("Big discount on the new SUV, book a test drive today").
The meaning expert says *Automotive* strongly; the keyword list recognises "SUV" and "टेस्ट ड्राइव"; nothing points
elsewhere. Result: **Automotive, strong match**.

**How sure is it?** When the top category clearly beats the others, the site shows *Strong match*. When it's closer,
*Good* or *Possible match*. Staff see the ads the system is least sure about on `/admin`, where they can correct them;
pressing **Retrain** teaches the system those corrections.

**How good is it?** We test it on ads from advertisers it has never seen: it gets **96 out of 100** right (98 out of 100 for
advertisers it knows), and 94 out of 100 on a separate set of Marathi/Hindi ads it was never trained on. See [Accuracy](#accuracy).

---

## 4. How it works (internal)

```
lokmattimes.com ─► Playwright driving Google Chrome ─► sponsored Taboola cards + Google ad frames
                                                        │ dedupe, strip tracking parameters
                                                        ▼
  multilingual sentence model (EmbeddingGemma) ► logistic regression ─┐ 75%
  TF-IDF word + character n-grams ───────────► logistic regression ─┘ 25%  ─► model score ─┐ 70%
  keyword lexicon per category (EN / मराठी / हिंदी) ─────────────────────────► rule score  ─┘ 30% (when a rule fires)
  advertiser memory (same advertiser seen before / corrected by staff) ───────► blended 50%
                                                        ▼
                     SQLite (data/ads.db) ─► Flask site / API / CSV ─► staff corrections ─► retrain
```

- **Collecting ads.** Lokmat Times injects its ads with JavaScript, so a plain HTTP request sees none. The collector opens
  each section in Google Chrome (hidden window), scrolls to trigger lazy loading, keeps only cards marked sponsored, and drops
  Lokmat's own article recommendations. Thumbnails are cached locally and served by the app.
- **Semantic model.** Google `EmbeddingGemma-300m` (100+ languages) turns an ad into a meaning vector, so
  "SUV", "car" and "गाडी" land close together even if the exact word never appeared in training. It runs locally on CPU
  via ONNX (`fastembed`); no data leaves the machine. The model file records which encoder it was trained with. If
  EmbeddingGemma can't be loaded, the app uses `paraphrase-multilingual-MiniLM-L12-v2`, then TF-IDF + rules.
  EmbeddingGemma is distributed under Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms), which allow
  commercial use subject to its prohibited-use policy; have legal glance at it before going live.
- **Marathi/Hindi keywords** must start a word (suffixes allowed: "फ्लॅटसाठी" matches "फ्लॅट"), so short words don't
  match inside longer ones.
- **Browser.** `launch_browser()` (app/scraper.py) starts **Google Chrome only** (`channel="chrome"`). If Chrome can't
  be started it raises a clear "install Google Chrome" error. (Playwright's own downloaded browser was dropped: Windows
  Smart App Control blocked it intermittently; Chrome is signed by Google and allowed.)
- **Live Site.** `app/live.py` `Snapshotter` opens the page headless, scrolls it like a visitor, labels each ad in place
  (outline + badge, reusing the library's category when the ad is known so both views agree), records each ad's
  position, and saves one image. The browser page overlays a highlight at those positions when an ad is clicked.
  `LiveView` does the same in a visible window for the staff demo. Image-only banners have no readable text and stay unlabelled.
- **Advertiser memory.** Advertisers rarely change category, so once an advertiser is known its label is blended in.
  Search-ad networks (srchly.co and similar) carry many unrelated advertisers, so there the advertiser *name* is the key;
  their links go to search-result pages, so the customer site shows those headlines without a link.

### Training data (`data/training_data.csv`, 684 rows)
- **382 real Lokmat Times ads**, hand-labelled (collected 23 and 24 Sep 2026).
- **221 curated examples** for categories that are rare in the live sample but common in Indian news advertising
  (jobs, government notices, education, automotive, real estate, food…).
- **81 curated Marathi/Hindi examples**, worded differently from the Marathi/Hindi test set (two that were too close to
  test items were reworded before measuring).

### Accuracy

Every row compares versions **on the same examples**, so the numbers are like-for-like.

| Measure | Round 1 | Round 2 (MiniLM) | **Round 5 (now)** |
|---|---|---|---|
| **New advertisers** (5-fold GroupKFold: an advertiser is never in both train and test), 684 examples | — | 89.6% | **96.1%** |
| Macro F1, same test | — | 0.90 | **0.965** |
| **New ads from known advertisers** (5-fold random split, with advertiser memory), 684 examples | — | 94.9% | **98.1%** |
| New advertisers, the 603-example set used before the Marathi/Hindi additions | — | 91.5% | **95.4%** |
| New advertisers, the 557-example set of round 2 | 80.4% | 92.3% | — |
| **Marathi/Hindi test set** (63 ads never used in training) | — | 79.4% | **93.7%** |

How to read this:
- **96.1%** is the headline figure. It's slightly flattered by the 81 easy curated Marathi/Hindi examples; on the
  real-ads-heavy 603-example set the new model scores **95.4%**.
- Weakest categories now: Entertainment & Media (F1 0.84; celebrity-news ads look like Sponsored Stories), then Fashion,
  Health and Home & Living (0.92–0.93). The 4 Marathi/Hindi misses: two Marathi real-estate headlines, a biryani offer and
  a packers-and-movers ad.
- Caveats: keyword lists were refined after looking at errors, and the known-advertiser figure benefits from
  near-duplicate creatives. Expect somewhat lower accuracy on unfamiliar kinds of ads. The `/admin` correction →
  retrain loop is how it keeps improving on real traffic.
- Model comparison behind the switch (same test, 557 examples): MiniLM 92.3% · mpnet-base 93.9% · Qwen3-0.6B 93.9% ·
  e5-large 94.8% · **EmbeddingGemma 95.7%**. Speed per ad: 25 / 62 / 748 / 218 / 79 ms.

---

## 5. Project layout

```
ad_categorizer/
├── run.py                 # start the server (Waitress); HOST / PORT / ADMIN_PASSWORD / REFRESH_HOURS
├── cli.py                 # command-line interface
├── setup.bat / run.bat    # Windows one-click setup & launch
├── requirements.txt
├── app/
│   ├── scraper.py         # Playwright collector for lokmattimes.com
│   ├── classifier.py      # categories, keyword rules, semantic + lexical models, evaluation
│   ├── softmax.py         # logistic regression in NumPy (avoids a DLL Smart App Control can block)
│   ├── live.py            # Live Site capture (Snapshotter) and staff demo window (LiveView)
│   ├── db.py              # SQLite storage
│   ├── web.py             # routes, staff protection, public API, auto refresh, image cache
│   ├── templates/         # overview, ad library, live site, categorize, staff, error pages, _ad_card
│   └── static/            # style.css, app.js (customers), admin.js (staff page only)
├── data/
│   ├── training_data.csv  # labelled training set
│   ├── sample_ads.json    # 411 ads collected 23–24 Sep 2026 (loaded on first run)
│   ├── eval_marathi_hindi.csv  # 63 Marathi/Hindi test ads, never used for training
│   └── ads.db, images/, snapshots/   # created at runtime
├── models/                # trained model + language model (~1.2 GB, created at setup)
└── tests/test_app.py      # 38 pytest tests, incl. "customers see no internals", staff protection, Marathi/Hindi accuracy
```

Run the tests with `python -m pytest`.

---

## 6. Limitations and notes

- **Which ads appear depends on the ad servers.** They choose ads per visitor, location and time; the server's location
  decides what the Live Site shows.
- **Image-only banners** can't be categorised from text, so they aren't labelled.
- **If Lokmat Times or its ad widgets change their HTML**, update the selectors in `TABOOLA_JS` (`app/scraper.py`) and
  `SCAN_JS` (`app/live.py`).
- **Load:** each Live Site capture opens one headless browser for ~30 s; captures run one at a time and are cached for
  10 minutes per section, so many customers viewing the same page cost one capture.
- Collection reads public pages only (about 10 page loads per run, every 6 hours by default). Check Lokmat Times' terms of
  use for your usage before going live.

### Troubleshooting

| Problem | Fix |
|---|---|
| `python` not recognised | Reinstall Python with "Add to PATH" ticked, or use `py` instead of `python`. |
| Live Site says "couldn't load the live page" / collection failed | See the server console. If it says *Google Chrome is needed*: install Chrome from https://www.google.com/chrome/ and restart the app. |
| `/admin` says "Page not found" | You're not on the server PC and `ADMIN_PASSWORD` isn't set. Set it and restart. |
| Language model download failed | The app still works (TF-IDF + rules). Connect to the internet and run `.venv\Scripts\python cli.py train`. |
| "Windows blocked a Python component… Application Control policy" | Windows Smart App Control / company policy is blocking a freshly downloaded, unsigned Python file. The app already avoids the one file seen blocked in testing. If another is blocked: run `setup.bat` again, ask IT to allow the folder, or use Python 3.12. |
| Port 5000 busy | `set PORT=8000` then `run.bat` |
| Want a clean start | Delete `data\ads.db`; the sample ads reload on next launch. |
| Model error after upgrading packages | Delete `models\ad_classifier.joblib`; it rebuilds automatically. |
