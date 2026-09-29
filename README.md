# Sortify — Lokmat Times Ad Categorizer

A simple web app that reads advertisements from **Lokmat Times** and automatically sorts them into 21 categories like **Real Estate**, **Cars**, **Education**, **Jobs**, and **Health**.

It understands ads written in **English, Marathi, and Hindi**.

---

## What Does It Do?

When you open the web app, you get 4 simple pages:

1. **Overview (`/`):** A clean dashboard showing how many ads were collected, top categories, and newest ads.
2. **Ad Library (`/ads`):** A gallery where you can search ads, look at their pictures, and filter by category (like Automotive or Health).
3. **Live Site (`/live`):** Shows the actual Lokmat Times website with each ad outlined and labelled in real time.
4. **Categorize an Ad (`/classify`):** A playground tool where you can type any ad headline or click examples, and the AI tells you its category on the spot.

---

## How to Run It (Step by Step)

### What You Need:
- **Python 3.10 or newer** (make sure *"Add Python to PATH"* was ticked when installing).
- **Google Chrome** installed on your computer.

---

### Step 1: Open your terminal
Open PowerShell or Command Prompt inside the project folder.

### Step 2: Create a virtual environment
This keeps all the project files and packages neat in one place:
```bash
python -m venv .venv
```

Now turn it on:
- **On Windows:**
  ```powershell
  .venv\Scripts\activate
  ```
- **On macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### Step 3: Install the packages
```bash
pip install -r requirements.txt
```

### Step 4: Train the AI model
This teaches the model how to recognize ad categories (it downloads the language model on the first run):
```bash
python cli.py train
```

### Step 5: Start the app!
```bash
python run.py
```

Your browser will automatically open at:
👉 **http://127.0.0.1:5000**

*(To stop the server, simply press `Ctrl + C` in your terminal window).*

---

## How the Categorizer Works (In Simple Words)

Think of it like sorting mail:

1. **It reads the text:** Looks at the headline, description, and website link of the ad.
2. **It understands languages:** It can read English, Marathi (मराठी), and Hindi (हिंदी).
3. **It looks for clues:**
   - Keywords (like *"BHK"* or *"flat"* → Real Estate, *"SUV"* → Automotive, *"MBBS"* → Education).
   - Word meaning (even if exact keywords aren't there, it understands the topic).
   - Known advertisers (if it knows *Policybazaar*, it remembers that's Insurance).
4. **It gives the verdict:** It shows the best category and how confident it is (*Strong match*, *Good match*, or *Possible match*).

---

## Project Folder Guide

```
Sortify--Smart_Ad_Categorizer/
├── app/
│   ├── classifier.py      # The brain: categorizes ads using machine learning
│   ├── scraper.py         # Visits Lokmat Times and collects live ads
│   ├── live.py            # Captures newspaper page screenshots and labels ads
│   ├── db.py              # Saves collected ads into a local SQLite database
│   ├── web.py             # Web pages, routes, and API endpoints
│   ├── templates/         # HTML pages for the website
│   └── static/            # CSS styling and JavaScript
├── data/
│   ├── training_data.csv  # Example ads used to train the model
│   ├── sample_ads.json    # Starter ads loaded on first launch
│   ├── eval_marathi_hindi.csv  # Marathi & Hindi test ads
│   └── images/            # Ad thumbnail pictures
├── models/                # Saved trained model file
├── tests/                 # Automated tests
├── cli.py                 # Helper commands
├── run.py                 # Starts the web app
├── requirements.txt       # List of Python libraries needed
└── pytest.ini             # Test runner setup
```

---

## Running the Tests

To make sure everything is working properly:
```bash
pytest
```
This runs the automated test suite to confirm all routes, categories, and models are working.

---

## Scan Full Page (rebuilt 29 Sep 2026)

**What it does:** upload a photo/scan of a newspaper page; Sortify finds the **advertisements only** (news is left out), reads each one, categorizes it, and shows a verification table with a picture of every ad, a numbered box on the page, a *calibrated* confidence, and a reading-quality score. **Download Verification Report** gives the table as a spreadsheet (.csv) with extra columns: advertiser, reading quality %, how it was read, position on page.

**How it works (plain words)**
1. **Find ad areas from the layout** (`app/page_scan.py`, `app/layout_features.py`): a small model looks at colour, pictures, box borders and lettering size, which is how ads differ from columns of news text. Trained on the pages in `test_scans/` (Lokmat + Lokmat Times, Marathi + English), at three quality levels (sharp, low-quality forward, blurry phone photo).
2. **Separate neighbouring ads** only along printed dividing lines, or white gaps where the two sides also differ in colour.
3. **Read each ad on its own**, enlarged and sharpened (much more accurate than reading the whole page at once). The headline is the biggest lettering, not simply the first line.
4. **Double-check for news wording** and drop empty or nested pieces.
5. **Categorize** with the normal Sortify model. **Confidence is now calibrated** on held-out data: this holds for clean text (the category model); scanned text is often garbled, so scan confidence is low.

**AI mode (optional, best accuracy):** if Anthropic credentials are present, pages are read by Claude vision instead (finds every ad, reads English/Marathi/Hindi print near-perfectly). To turn it on: `setx ANTHROPIC_API_KEY your-key` and restart, or run `ant auth login`. Turn off with `SORTIFY_AI=off`. If AI is unavailable or fails, Sortify silently uses the local engine and says so in the report ("Read on this computer"). Page images are sent to Anthropic only in AI mode. *AI mode is covered by automated tests with a simulated response, but was not run against the real service on this PC (no key here).*

**Measured results of the local engine (final test, 16 pages / 21 ads, page never seen in training): NOT good enough yet.** Found 11 of 21 ads, reported 15 non-ads, right category on only 3 of the 11 found, average reading quality 0.23, average confidence 0.18. Main causes: the free on-computer text reader (EasyOCR) misreads Marathi and stylised ad lettering, so the text fed to the categorizer is wrong; and ads without strong colour or borders are missed. The earlier statement that confidence is "calibrated" holds only for typed/clean text (the category model itself: 95% accurate), not for garbled scan text. **For real use, turn on AI mode (above), which is designed to fix exactly these two problems, but it has not been tested against the real service on this PC.** Until then, treat local-engine scan results as a rough first draft and verify every row.

**Check it yourself:** run `.venv\Scripts\python tools\eval_scan.py --cv -v` (scores the scanner against the hand-marked answer key `test_scans/ground_truth.json`, 16 pages / 21 ads). Run `.venv\Scripts\python -m pytest`.
