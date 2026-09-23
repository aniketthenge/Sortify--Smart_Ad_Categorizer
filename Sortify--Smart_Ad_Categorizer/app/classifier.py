"""
Hybrid ad classifier.

1. Machine-learning model - TF-IDF over word n-grams (1-2) and character
   n-grams (2-5, which also copes with Marathi/Hindi text and brand-name
   fragments like "matrimony" inside "elitematrimony.com") feeding a
   class-balanced Logistic Regression.
2. Keyword rules - a curated lexicon per category. Rule hits are blended into
   the model probabilities, which helps on short or unusual ads the model has
   never seen, and gives a human-readable reason for each decision.

Each prediction returns: category, confidence, top-3 alternatives, matched
keywords, and the most influential model terms.
"""
from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parent.parent
TRAINING_CSV = ROOT / "data" / "training_data.csv"
MODEL_PATH = ROOT / "models" / "ad_classifier.joblib"

ML_WEIGHT = 0.75          # share of the final score coming from the model
REVIEW_THRESHOLD = 0.35   # below this the ad is flagged for human review

# name -> (colour for UI, keyword lexicon). Keywords are matched as whole
# words / phrases, case-insensitive; Devanagari keywords are matched as substrings.
CATEGORIES: dict[str, tuple[str, list[str]]] = {
    "Finance & Investment": ("#1f77b4", ["trading", "trader", "intraday", "forex", "stock market", "share market", "demat",
        "mutual fund", "sip", "small cap", "fixed deposit", "fd", "loan", "emi", "credit card", "invest", "investment",
        "crypto", "bitcoin", "brokerage", "wealth", "finance", "bank", "शेअर", "गुंतवणूक", "कर्ज"]),
    "Insurance": ("#17becf", ["insurance", "term plan", "term insurance", "life cover", "policy", "ulip", "premium",
        "cashless", "claim settlement", "pru life", "विमा"]),
    "Health & Wellness": ("#2ca02c", ["hearing aid", "hearing", "doctor", "clinic", "hospital", "treatment", "pain",
        "diabetes", "ayurvedic", "medicine", "dental", "ivf", "weight loss", "diet", "yoga", "flexibility", "stiff",
        "health checkup", "audiologist", "therapy", "रुग्णालय", "उपचार", "आरोग्य"]),
    "Matrimony & Dating": ("#e377c2", ["matrimony", "matchmaking", "life partner", "bride", "groom", "marriage bureau",
        "dating", "singles", "shaadi", "वधू", "वर"]),
    "Gaming": ("#9467bd", ["game", "gaming", "play free", "play now", "mmo", "shooter", "rummy", "fantasy cricket",
        "ludo", "players", "rpg", "war thunder", "crossout"]),
    "Education & Courses": ("#8c564b", ["course", "courses", "admission", "school", "college", "university", "mba",
        "upsc", "mpsc", "coaching", "classes", "learn", "public speaking", "certification", "degree", "परीक्षा", "शिक्षण"]),
    "Real Estate": ("#bcbd22", ["apartments", "flats", "bhk", "plots", "villa", "new launch", "possession", "rera",
        "township", "pre-launch", "property for sale", "real estate", "घरे", "फ्लॅट"]),
    "Home & Living": ("#7f7f7f", ["furniture", "sofa", "mattress", "bedding", "sheets", "kitchen", "interior", "mosaic",
        "tiles", "paint", "bath", "shower", "remodel", "office pods", "home decor", "murals"]),
    "Fashion & Shopping": ("#d62728", ["saree", "sarees", "blouses", "tops", "outfits", "styles", "fashion", "wear",
        "kurta", "shoes", "jewellery", "jewelry", "gold ring", "sale", "off", "gear", "gifts", "collection", "कपडे", "सूट"]),
    "Technology & Electronics": ("#ff7f0e", ["cloud", "server", "servers", "hosting", "ddos", "ai", "software",
        "cybersecurity", "data breach", "smartphone", "laptop", "broadband", "5g", "gpu", "xeon", "solar generator",
        "earbuds", "smart tv", "threat detection"]),
    "Automotive": ("#393b79", ["car", "cars", "suv", "sedan", "bike", "scooter", "ev", "electric vehicle", "test drive",
        "motors", "ex90", "mileage", "स्कूटर", "गाडी"]),
    "Travel & Tourism": ("#637939", ["travel", "tour", "trip", "flights", "hotel", "resort", "vacation", "vacations",
        "campervan", "private jet", "holiday", "yatra", "beaches", "पर्यटन"]),
    "Charity & NGO": ("#8c6d31", ["donate", "donation", "help him", "help her", "transplant", "save a life", "ngo",
        "foundation", "orphan", "orphans", "orphanage", "annadaan", "support homeless", "80g", "दान"]),
    "Astrology & Spirituality": ("#843c39", ["numerology", "astrology", "astrologer", "horoscope", "kundli", "vastu",
        "vedic", "puja", "zodiac", "birth date", "gemstone", "कुंडली", "भविष्य"]),
    "Industrial & B2B": ("#7b4173", ["industry", "industrial", "manufacturing", "machine", "machines", "machine tool",
        "aerospace", "precision engineering", "diesel generator", "container", "enterprise", "supplier", "b2b",
        "production", "wholesale", "cat®"]),
    "Food & Beverage": ("#e7ba52", ["coffee", "food", "restaurant", "biryani", "groceries", "snacks", "masala", "rice",
        "tea", "ice cream", "recipe", "flavor", "flavour"]),
    "Entertainment & Media": ("#ce6dbd", ["movie", "film", "web series", "streaming", "watch", "concert", "music",
        "youtube", "ott", "tmz", "episode", "चित्रपट"]),
    "Professional Services": ("#6b6ecf", ["lawyer", "lawyers", "legal", "attorney", "gst", "itr", "tax filing",
        "registration", "packers", "movers", "cremation", "funeral", "pest control", "consultant"]),
    "Jobs & Careers": ("#9edae5", ["jobs", "job", "hiring", "vacancy", "walk-in", "recruitment", "bharti", "freshers",
        "career", "salary", "work from home", "नोकरी", "भरती"]),
    "Government & Public Notices": ("#aec7e8", ["government", "municipal", "corporation", "tender", "notice", "yojana",
        "ministry", "public notice", "e-auction", "sarfaesi", "election", "gov.in", "सूचना", "शासन"]),
    "Sponsored Content / Clickbait": ("#c7c7c7", ["you won't believe", "will surprise you", "take a look inside",
        "what happens next", "what happened next", "hold on tight", "see what", "shocking", "stun you", "incredible",
        "celebrities", "celebrity", "most beautiful", "[gallery]", "[see list]", "jaw-dropping", "look closely",
        "remember him", "what he looks like now", "what she did next", "unbelievable", "the result was"]),
}
CATEGORY_NAMES = list(CATEGORIES)


# ---------------------------------------------------------------- features
def build_text(title: str = "", description: str = "", advertiser: str = "", domain: str = "") -> str:
    """Combine ad fields into one document. The domain is split into words so
    'elitematrimony.com' and 'tradewise.thefuture.university' contribute tokens."""
    domain_tokens = " ".join(t for t in re.split(r"[.\-_/]+", (domain or "").lower())
                             if t and t not in {"www", "com", "in", "co", "net", "org"})
    return " ".join(p for p in [title, description, advertiser, domain_tokens] if p).strip()


def _new_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True,
                                 token_pattern=r"(?u)\b\w[\w'®]+\b", sublinear_tf=True, min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True,
                                 sublinear_tf=True, min_df=1)),
    ])
    clf = LogisticRegression(C=8.0, max_iter=3000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


# ---------------------------------------------------------------- rules
_KW_PATTERNS: dict[str, list[tuple[str, re.Pattern]]] = {}
for _cat, (_c, _kws) in CATEGORIES.items():
    pats = []
    for kw in _kws:
        if re.search(r"[ऀ-ॿ]", kw):  # Devanagari -> substring match
            pats.append((kw, re.compile(re.escape(kw))))
        else:
            pats.append((kw, re.compile(r"(?<![\w])" + re.escape(kw) + r"(?![\w])", re.I)))
    _KW_PATTERNS[_cat] = pats


def keyword_scores(text: str) -> tuple[np.ndarray, dict[str, list[str]]]:
    """Return a probability-like vector over CATEGORY_NAMES and the matched keywords."""
    hits: dict[str, list[str]] = {}
    scores = np.zeros(len(CATEGORY_NAMES))
    for i, cat in enumerate(CATEGORY_NAMES):
        matched = [kw for kw, p in _KW_PATTERNS[cat] if p.search(text)]
        if matched:
            hits[cat] = matched
            scores[i] = sum(1.0 + 0.5 * (len(k.split()) - 1) for k in matched)  # phrases weigh more
    total = scores.sum()
    return (scores / total if total else scores), hits


# ---------------------------------------------------------------- data
def load_training_rows(extra_rows: list[dict] | None = None) -> list[dict]:
    rows = []
    with open(TRAINING_CSV, encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))
    for r in extra_rows or []:
        rows.append({**r, "source": r.get("source", "user_feedback")})
    return [r for r in rows if r.get("category") in CATEGORIES and (r.get("title") or "").strip()]


# ---------------------------------------------------------------- model
class AdClassifier:
    def __init__(self, pipeline: Pipeline | None = None, meta: dict | None = None):
        self.pipeline = pipeline
        self.meta = meta or {}

    # -- training ---------------------------------------------------------
    @classmethod
    def train(cls, extra_rows: list[dict] | None = None, evaluate: bool = True) -> "AdClassifier":
        rows = load_training_rows(extra_rows)
        X = [build_text(r["title"], r.get("description", ""), r.get("advertiser", ""), r.get("domain", "")) for r in rows]
        y = [r["category"] for r in rows]
        meta = {
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "n_samples": len(rows),
            "n_feedback": sum(1 for r in rows if r.get("source") == "user_feedback"),
            "class_counts": {c: y.count(c) for c in CATEGORY_NAMES},
        }
        if evaluate:
            meta["evaluation"] = cls._cross_validate(X, y, [r.get("domain") or r["title"] for r in rows])
        pipe = _new_pipeline().fit(X, y)
        model = cls(pipe, meta)
        model.save()
        return model

    @staticmethod
    def _cross_validate(X, y, groups, folds: int = 5) -> dict:
        """GroupKFold by advertiser domain: the same advertiser never appears in
        both train and test, so the score reflects performance on *new* advertisers."""
        X, y = np.array(X, dtype=object), np.array(y)
        y_true, y_ml, y_hybrid = [], [], []
        for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
            pipe = _new_pipeline().fit(X[tr], y[tr])
            tmp = AdClassifier(pipe)
            for text, label in zip(X[te], y[te]):
                y_true.append(label)
                y_ml.append(pipe.predict([text])[0])
                y_hybrid.append(tmp._predict_text(text)["category"])
        report = classification_report(y_true, y_hybrid, output_dict=True, zero_division=0)
        return {
            "method": f"{folds}-fold GroupKFold by advertiser domain",
            "ml_only_accuracy": round(accuracy_score(y_true, y_ml), 3),
            "accuracy": round(accuracy_score(y_true, y_hybrid), 3),
            "macro_f1": round(f1_score(y_true, y_hybrid, average="macro", zero_division=0), 3),
            "per_class": {c: {"precision": round(v["precision"], 2), "recall": round(v["recall"], 2),
                              "f1": round(v["f1-score"], 2), "support": int(v["support"])}
                          for c, v in report.items() if c in CATEGORIES},
        }

    # -- persistence ------------------------------------------------------
    def save(self, path: Path = MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "AdClassifier":
        if not path.exists():
            return cls.train()
        try:
            blob = joblib.load(path)
            return cls(blob["pipeline"], blob["meta"])
        except Exception:  # file from another scikit-learn version / corrupted -> rebuild
            return cls.train()

    # -- inference --------------------------------------------------------
    def _ml_proba(self, text: str) -> np.ndarray:
        proba = self.pipeline.predict_proba([text])[0]
        out = np.zeros(len(CATEGORY_NAMES))
        for cls_name, p in zip(self.pipeline.classes_, proba):
            out[CATEGORY_NAMES.index(cls_name)] = p
        return out

    def _top_terms(self, text: str, category: str, k: int = 6) -> list[str]:
        """Word features pushing the model towards `category` (for explainability)."""
        feats = self.pipeline.named_steps["features"]
        clf = self.pipeline.named_steps["clf"]
        if category not in clf.classes_:
            return []
        word_vec = feats.transformer_list[0][1]
        n_word = len(word_vec.vocabulary_)
        x = feats.transform([text]).tocsr()
        coef = clf.coef_[list(clf.classes_).index(category)]
        vocab_inv = {i: t for t, i in word_vec.vocabulary_.items()}
        contrib = [(x[0, j] * coef[j], vocab_inv[j]) for j in x.indices if j < n_word]
        return [t for v, t in sorted(contrib, reverse=True)[:k] if v > 0]

    def _predict_text(self, text: str) -> dict:
        ml = self._ml_proba(text)
        kw, hits = keyword_scores(text)
        combined = ML_WEIGHT * ml + (1 - ML_WEIGHT) * kw if hits else ml
        order = np.argsort(combined)[::-1]
        best = CATEGORY_NAMES[order[0]]
        conf = float(combined[order[0]])
        return {
            "category": best,
            "confidence": round(conf, 3),
            "method": "ml+rules" if hits else "ml",
            "needs_review": conf < REVIEW_THRESHOLD,
            "alternatives": [{"category": CATEGORY_NAMES[i], "score": round(float(combined[i]), 3)} for i in order[:3]],
            "matched_keywords": hits.get(best, []),
            "all_keyword_hits": hits,
            "ml_confidence": round(float(ml[order[0]]), 3),
        }

    def predict(self, title: str = "", description: str = "", advertiser: str = "", domain: str = "") -> dict:
        text = build_text(title, description, advertiser, domain)
        if not text:
            return {"category": "Sponsored Content / Clickbait", "confidence": 0.0, "method": "empty",
                    "needs_review": True, "alternatives": [], "matched_keywords": [], "top_terms": []}
        result = self._predict_text(text)
        result["top_terms"] = self._top_terms(text, result["category"])
        return result


def category_colors() -> dict[str, str]:
    return {name: color for name, (color, _) in CATEGORIES.items()}
