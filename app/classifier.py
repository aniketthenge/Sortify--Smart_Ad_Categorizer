"""
Hybrid ad classifier.

Three signals are combined into one probability per category:

1. Semantic model - a pretrained multilingual sentence encoder
   (Google EmbeddingGemma-300m, run locally via ONNX/fastembed; falls back to
   paraphrase-multilingual-MiniLM-L12-v2) turns the ad into a meaning vector,
   fed to Logistic Regression.
   It generalises to wording it has never seen ("SUV" ~ "car", Marathi ~ English).
2. Lexical model - TF-IDF word (1-2) and character (2-5) n-grams + Logistic
   Regression. Picks up brand fragments and exact phrases.
3. Keyword rules - a curated lexicon per category.

On top of that, an advertiser memory: if the same advertiser was labelled
before (training data or a staff correction), that label is blended in, since
advertisers almost always stay in one category.

If no semantic model can be loaded (no internet on first run, package
missing), the classifier falls back to lexical + rules automatically.
"""
from __future__ import annotations

import csv
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import FeatureUnion, Pipeline

from .softmax import SoftmaxRegression

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless on Windows

ROOT = Path(__file__).resolve().parent.parent
TRAINING_CSV = ROOT / "data" / "training_data.csv"
MODEL_PATH = ROOT / "models" / "ad_classifier.joblib"
EMBED_CACHE = ROOT / "models" / "embeddings"
# Sentence encoders in order of preference: (name, text prefix the model expects).
# EmbeddingGemma scored 95.4% vs 91.5% for MiniLM on unseen advertisers (README, "Accuracy").
EMBED_MODELS = [
    ("google/embeddinggemma-300m", "task: classification | query: "),                  # ~1.2 GB
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", ""),                # ~0.2 GB fallback
]

# Blend weights (tuned with grouped cross-validation, see _cross_validate).
W_SEMANTIC, W_LEXICAL = 0.75, 0.25   # inside the model score
W_MODEL = 0.7                        # model vs keyword rules, when a rule fires
W_MEMORY = 0.5                       # known advertiser vs everything else
REVIEW_THRESHOLD = 0.35              # below this an ad is flagged for staff review

# Search-arbitrage networks show ads for many unrelated advertisers under one
# domain, so their domain says nothing about the category.
NETWORK_DOMAINS = {"srchly.co", "searchinfohub.com", "searchresultoffers.com"}

# name -> (colour for UI, keyword lexicon). Keywords are matched as whole
# words / phrases, case-insensitive; non-Latin keywords as substrings.
CATEGORIES: dict[str, tuple[str, list[str]]] = {
    "Finance & Investment": ("#1d4ed8", ["trading", "trader", "traders", "intraday", "forex", "stock market",
        "share market", "stocks", "demat", "mutual fund", "mutual funds", "sip", "small cap", "small caps",
        "fixed deposit", "fd", "loan", "loans", "personal loan", "home loan", "emi", "credit card", "invest",
        "investing", "investment", "crypto", "bitcoin", "brokerage", "wealth", "finance", "bank", "savings account",
        "interest rate", "returns", "portfolio", "gold bond", "market profile", "side income", "passive income",
        "शेअर", "गुंतवणूक", "कर्ज", "बँक", "निवेश", "लोन"]),
    "Insurance": ("#0e7490", ["insurance", "insure", "term plan", "term insurance", "life cover", "policy",
        "ulip", "premium", "cashless", "claim settlement", "pru life", "health cover", "sum assured",
        "विमा", "बीमा"]),
    "Health & Wellness": ("#15803d", ["hearing aid", "hearing aids", "hearing", "doctor", "doctors", "clinic",
        "hospital", "treatment", "pain", "diabetes", "ayurvedic", "medicine", "medicines", "dental", "implants",
        "ivf", "weight loss", "diet", "yoga", "flexibility", "stiff", "health checkup", "audiologist", "therapy",
        "healing", "trauma", "surgery", "knee", "joint", "eye", "lenses", "nutrition", "fitness", "wellness",
        "रुग्णालय", "उपचार", "आरोग्य", "इलाज", "डॉक्टर"]),
    "Matrimony & Dating": ("#be185d", ["matrimony", "matrimonial", "matchmaking", "life partner", "bride",
        "brides", "groom", "grooms", "marriage bureau", "dating", "singles", "shaadi", "soulmate", "relationship",
        "relationships", "वधू", "वर", "विवाह", "रिश्ते"]),
    "Gaming": ("#6d28d9", ["game", "games", "gaming", "play free", "play now", "free to play", "mmo", "shooter",
        "rummy", "fantasy cricket", "ludo", "players", "rpg", "war thunder", "crossout", "multiplayer", "esports",
        "tanks", "battle"]),
    "Education & Courses": ("#b45309", ["course", "courses", "admission", "admissions", "school", "college",
        "university", "mba", "masters", "degree", "diploma", "pg diploma", "upsc", "mpsc", "neet", "jee",
        "coaching", "classes", "tuition", "learn", "learning", "public speaking", "certification", "certified",
        "skills", "students", "batch", "scholarship", "study", "परीक्षा", "शिक्षण", "प्रवेश", "पढ़ाई"]),
    "Real Estate": ("#92400e", ["apartments", "apartment", "flats", "flat", "bhk", "plots", "plot", "villa",
        "villas", "new launch", "possession", "rera", "township", "pre-launch", "property for sale", "real estate",
        "homes in", "residences", "office space", "site visit", "ready to move", "घरे", "फ्लॅट", "फ्लैट", "प्लॉट"]),
    "Home & Living": ("#0f766e", ["furniture", "sofa", "mattress", "bedding", "sheets", "kitchen", "interior",
        "interiors", "mosaic", "tiles", "paint", "paints", "bath", "bathroom", "shower", "remodel", "remodeling",
        "renovation", "renovations", "office pods", "home decor", "murals", "curtains", "faucets", "ac",
        "air conditioner", "water purifier", "appliances", "chimney", "waterproofing", "your space"]),
    "Fashion & Shopping": ("#e11d48", ["saree", "sarees", "blouses", "tops", "outfits", "outfit", "styles",
        "fashion", "wear", "ethnic wear", "kurta", "kurtas", "shoes", "sneakers", "jewellery", "jewelry", "gold ring",
        "ring", "sale", "off", "gear", "gifts", "gift", "gifting", "collection", "clothing", "jacket", "jackets",
        "linen", "fabric", "insulation", "stormproof", "waterproof", "down", "watches", "shopping", "deals",
        "कपडे", "कपड़", "साडी", "साड्या", "खरेदी", "दागिने"]),
    "Technology & Electronics": ("#c2410c", ["cloud", "server", "servers", "hosting", "ddos", "ai",
        "artificial intelligence", "software", "saas", "app", "platform", "cybersecurity", "cyber", "threat",
        "data breach", "smartphone", "phone", "laptop", "laptops", "broadband", "5g", "gpu", "xeon", "kubernetes",
        "solar generator", "earbuds", "smart tv", "smart glasses", "glasses", "threat detection", "ecommerce",
        "website development", "shopify", "crm", "touch screens", "displays", "compliance", "workforce management",
        "feedback platform", "automation software"]),
    "Automotive": ("#0e5fa8", ["car", "cars", "suv", "suvs", "sedan", "hatchback", "bike", "bikes", "motorcycle",
        "scooter", "ev", "electric vehicle", "test drive", "motors", "ex90", "ex40", "mileage", "driving", "tyre",
        "tire", "roadside", "horsepower", "on-road price", "स्कूटर", "गाडी", "कार", "बाइक"]),
    "Travel & Tourism": ("#0369a1", ["travel", "tour", "tours", "trip", "trips", "flights", "flight", "hotel",
        "hotels", "resort", "vacation", "vacations", "holiday", "holidays", "campervan", "camper", "rv", "cruise",
        "cruises", "private jet", "yatra", "beaches", "visa", "car rental", "sightseeing", "getaway",
        "पर्यटन", "सहल", "यात्रा"]),
    "Charity & NGO": ("#047857", ["donate", "donation", "donations", "help him", "help her", "transplant",
        "save a life", "save her", "save him", "ngo", "foundation", "orphan", "orphans", "orphanage", "annadaan",
        "support homeless", "homeless mothers", "80g", "children in need", "feed a child", "contribution",
        "fundraiser", "दान", "अन्नदान", "मदत करा", "ದಾನ", "ನೆರವ"]),
    "Astrology & Spirituality": ("#4f46e5", ["numerology", "numerologist", "astrology", "astrologer", "horoscope",
        "kundli", "vastu", "vedic", "puja", "zodiac", "birth date", "gemstone", "temple", "पूजा", "कुंडली",
        "भविष्य", "ज्योतिष"]),
    "Industrial & B2B": ("#57534e", ["industry", "industrial", "manufacturing", "manufacturer", "machine",
        "machines", "machine tool", "machinery", "aerospace", "precision engineering", "precision", "diesel generator",
        "diesel generators", "generators", "container", "enterprise", "supplier", "suppliers", "supplies", "b2b",
        "production", "wholesale", "cat®", "linear motion", "lab equipment", "cleaning equipment", "equipment",
        "logistics", "freight", "factories", "business continuity", "automation"]),
    "Food & Beverage": ("#4d7c0f", ["coffee", "food", "restaurant", "restaurants", "biryani", "pizza", "burger",
        "groceries", "snacks", "masala", "rice", "tea", "ice cream", "recipe", "flavor", "flavour", "ghee", "oil",
        "dry fruits", "order online", "फराळ", "खाद्य"]),
    "Entertainment & Media": ("#a21caf", ["movie", "movies", "film", "films", "web series", "streaming", "stream",
        "concert", "music", "songs", "youtube", "ott", "tmz", "episode", "episodes", "trailer", "tickets",
        "comedy show", "news", "चित्रपट", "फिल्म"]),
    "Professional Services": ("#7c2d12", ["lawyer", "lawyers", "legal", "attorney", "injury", "claims", "gst",
        "itr", "tax filing", "registration", "packers", "movers", "cremation", "funeral", "pest control",
        "consultant", "consultation", "chartered accountant", "company registration"]),
    "Jobs & Careers": ("#b91c1c", ["jobs", "job", "hiring", "vacancy", "vacancies", "walk-in", "recruitment",
        "bharti", "freshers", "career", "careers", "salary", "work from home", "apply now", "interview",
        "नोकरी", "भरती", "नौकरी", "भर्ती"]),
    "Government & Public Notices": ("#1e3a8a", ["government", "municipal", "corporation", "tender", "notice",
        "yojana", "ministry", "public notice", "e-auction", "sarfaesi", "election", "gov.in", "aadhaar", "scheme",
        "department", "सूचना", "शासन", "सरकार", "योजना"]),
    "Sponsored Stories": ("#64748b", ["you won't believe", "will surprise you", "might surprise you",
        "take a look inside", "take a peek inside", "look inside", "what happens next", "what happened next",
        "what happens", "watch what", "hold on tight", "see what", "shocking", "stun you", "incredible",
        "celebrities", "celebrity", "celebs", "stars who", "most beautiful", "[gallery]", "[pics]", "[story]",
        "[see list]", "[view photos]", "[see more]", "[view now]", "jaw-dropping", "look closely", "remember him",
        "looks like now", "what he looks like", "what she did next", "unbelievable", "the result was",
        "here's why", "heres why", "here's what", "everyone should know", "people should know", "you should know",
        "trick", "then and now", "wait till you see", "before you see", "no one was supposed", "the reason",
        "can tell about you", "personality", "stepson", "mother-in-law", "husband", "wife", "zookeeper", "vet",
        "reunites", "glow-ups", "transformations", "the neighbor", "manager's", "hollywood"]),
}
CATEGORY_NAMES = list(CATEGORIES)
_IDX = {c: i for i, c in enumerate(CATEGORY_NAMES)}


# ---------------------------------------------------------------- features
def build_text(title: str = "", description: str = "", advertiser: str = "", domain: str = "") -> str:
    """Combine ad fields into one document. The domain is split into words so
    'elitematrimony.com' and 'tradewise.thefuture.university' contribute tokens."""
    domain_tokens = " ".join(t for t in re.split(r"[.\-_/]+", (domain or "").lower())
                             if t and t not in {"www", "com", "in", "co", "net", "org"})
    return " ".join(p for p in [title, description, advertiser, domain_tokens] if p).strip()


def advertiser_key(advertiser: str = "", domain: str = "") -> str:
    """Identity of the advertiser: the domain, except on search-ad networks where
    the advertiser name is the only thing that identifies who is advertising."""
    domain = (domain or "").lower()
    if domain.startswith("www."):
        domain = domain[4:]
    if domain in NETWORK_DOMAINS or not domain:
        name = re.sub(r"\s*\|.*$", "", (advertiser or "")).strip().lower()
        return f"{domain}:{name}"
    return domain


def _lexical_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True,
                                 token_pattern=r"(?u)\b\w[\w'®]+\b", sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True, sublinear_tf=True)),
    ])
    return Pipeline([("features", features), ("clf", SoftmaxRegression(C=8.0))])


def _semantic_clf() -> SoftmaxRegression:
    return SoftmaxRegression(C=30.0)


# ---------------------------------------------------------------- embeddings
class _Encoder:
    """Lazy singleton around the ONNX sentence encoder (downloaded once into models/embeddings)."""
    _model = None
    name: str | None = None
    prefix = ""
    _tried: set = set()

    @classmethod
    def available(cls, name: str | None = None) -> bool:
        """Load `name` (or the best one that works). Returns True if an encoder is ready."""
        if cls._model is not None and (name is None or name == cls.name):
            return True
        for model_name, prefix in EMBED_MODELS:
            if (name and model_name != name) or model_name in cls._tried:
                continue
            cls._tried.add(model_name)
            try:
                from fastembed import TextEmbedding
                cls._model = TextEmbedding(model_name, cache_dir=str(EMBED_CACHE))
                cls.name, cls.prefix = model_name, prefix
                return True
            except Exception as exc:  # offline first run, blocked download, missing package
                print(f"[classifier] could not load {model_name}: {exc}")
        if cls._model is None:
            print("[classifier] no semantic model available, using lexical model only")
        return False

    @classmethod
    def encode(cls, texts: list[str]) -> np.ndarray:
        E = np.array(list(cls._model.embed([cls.prefix + str(t) for t in texts])), dtype=np.float32)
        return E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-9, None)


# ---------------------------------------------------------------- rules
_KW_PATTERNS: dict[str, list[tuple[str, re.Pattern]]] = {}
for _cat, (_c, _kws) in CATEGORIES.items():
    pats = []
    for kw in _kws:
        if re.search(r"[^\x00-\x7F®]", kw):  # Devanagari / Kannada
            # Marathi/Hindi attach suffixes (घरे -> घरांसाठी), so allow text after the keyword, but the keyword
            # must start a word; otherwise short words like "वर" (groom) match inside "कपड्यांवर" / "मूवर्स".
            pats.append((kw, re.compile(r"(?<![\u0900-\u097F\u0C80-\u0CFF])" + re.escape(kw))))
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


def _memory_from(keys: list[str], labels: list[str]) -> dict[str, dict[str, float]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for k, lab in zip(keys, labels):
        counts[k][lab] += 1
    return {k: {lab: n / sum(c.values()) for lab, n in c.items()} for k, c in counts.items()}


# ---------------------------------------------------------------- model
class AdClassifier:
    def __init__(self, lexical: Pipeline | None = None, semantic: SoftmaxRegression | None = None,
                 memory: dict | None = None, meta: dict | None = None):
        self.lexical = lexical
        self.semantic = semantic
        self.memory = memory or {}
        self.meta = meta or {}

    # -- training ---------------------------------------------------------
    @classmethod
    def train(cls, extra_rows: list[dict] | None = None, evaluate: bool = True) -> "AdClassifier":
        rows = load_training_rows(extra_rows)
        X = [build_text(r["title"], r.get("description", ""), r.get("advertiser", ""), r.get("domain", "")) for r in rows]
        y = np.array([r["category"] for r in rows])
        keys = [advertiser_key(r.get("advertiser", ""), r.get("domain", "")) for r in rows]
        E = _Encoder.encode(X) if _Encoder.available() else None
        meta = {
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "n_samples": len(rows),
            "n_feedback": sum(1 for r in rows if r.get("source") == "user_feedback"),
            "semantic": E is not None,
            "embed_model": _Encoder.name if E is not None else None,
            "class_counts": {c: int((y == c).sum()) for c in CATEGORY_NAMES},
        }
        if evaluate:
            meta["evaluation"] = cls._cross_validate(np.array(X, dtype=object), y, E, keys)
        model = cls(_lexical_pipeline().fit(X, y), _semantic_clf().fit(E, y) if E is not None else None,
                    _memory_from(keys, list(y)), meta)
        model.save()
        return model

    @staticmethod
    def _cross_validate(X, y, E, keys, folds: int = 5) -> dict:
        """GroupKFold by advertiser: an advertiser is never in both train and test,
        so the score is the accuracy on advertisers the model has never seen."""
        def run(splits, use_memory):
            t, p = [], []
            for tr, te in splits:
                tmp = AdClassifier(_lexical_pipeline().fit(X[tr], y[tr]),
                                   _semantic_clf().fit(E[tr], y[tr]) if E is not None else None,
                                   _memory_from([keys[i] for i in tr], list(y[tr])) if use_memory else None)
                lex = tmp._proba(tmp.lexical, X[te])
                sem = tmp._proba(tmp.semantic, E[te]) if E is not None else None
                for i, j in enumerate(te):
                    t.append(y[j])
                    p.append(tmp._combine(X[j], lex[i], sem[i] if sem is not None else None, keys[j])["category"])
            return t, p

        y_true, y_pred = run(GroupKFold(n_splits=folds).split(X, y, keys), use_memory=False)
        # New ads from advertisers already seen (the everyday case): random split + advertiser memory.
        k_true, k_pred = run(KFold(n_splits=folds, shuffle=True, random_state=0).split(X), use_memory=True)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        return {
            "method": f"{folds}-fold GroupKFold by advertiser",
            "accuracy": round(accuracy_score(y_true, y_pred), 3),
            "known_advertiser_accuracy": round(accuracy_score(k_true, k_pred), 3),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3),
            "per_class": {c: {"precision": round(v["precision"], 2), "recall": round(v["recall"], 2),
                              "f1": round(v["f1-score"], 2), "support": int(v["support"])}
                          for c, v in report.items() if c in CATEGORIES},
        }

    # -- persistence ------------------------------------------------------
    def save(self, path: Path | None = None):
        path = path or MODEL_PATH  # looked up at call time, so tests/tools can redirect it
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"version": 2, "lexical": self.lexical, "semantic": self.semantic,
                     "memory": self.memory, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: Path | None = None) -> "AdClassifier":
        path = path or MODEL_PATH
        if not path.exists():
            return cls.train()
        try:
            blob = joblib.load(path)
            if blob.get("version") != 2:
                return cls.train()
            model = cls(blob["lexical"], blob["semantic"], blob["memory"], blob["meta"])
            if model.semantic is not None and not _Encoder.available(model.meta.get("embed_model")):
                model.semantic = None  # the encoder it was trained with can't be loaded: degrade gracefully
            return model
        except Exception:  # file from another scikit-learn version / corrupted -> rebuild
            return cls.train()

    def remember(self, advertiser: str, domain: str, category: str):
        """Staff correction: remember this advertiser's category immediately."""
        self.memory[advertiser_key(advertiser, domain)] = {category: 1.0}

    # -- inference --------------------------------------------------------
    @staticmethod
    def _proba(clf, inputs) -> np.ndarray:
        raw = clf.predict_proba(inputs)
        out = np.zeros((raw.shape[0], len(CATEGORY_NAMES)))
        for j, c in enumerate(clf.classes_):
            out[:, _IDX[c]] = raw[:, j]
        return out

    def _combine(self, text: str, lex: np.ndarray, sem: np.ndarray | None, key: str | None = None) -> dict:
        model = W_SEMANTIC * sem + W_LEXICAL * lex if sem is not None else lex
        kw, hits = keyword_scores(text)
        score = W_MODEL * model + (1 - W_MODEL) * kw if hits else model
        known = self.memory.get(key) if key else None
        if known:
            mem = np.zeros(len(CATEGORY_NAMES))
            for c, p in known.items():
                mem[_IDX[c]] = p
            score = (1 - W_MEMORY) * score + W_MEMORY * mem
        order = np.argsort(score)[::-1]
        conf = float(score[order[0]])
        return {
            "category": CATEGORY_NAMES[order[0]],
            "confidence": round(conf, 3),
            "method": "+".join(["semantic" if sem is not None else "lexical"] + (["rules"] if hits else [])
                               + (["advertiser"] if known else [])),
            "needs_review": conf < REVIEW_THRESHOLD,
            "alternatives": [{"category": CATEGORY_NAMES[i], "score": round(float(score[i]), 3)} for i in order[:3]],
            "matched_keywords": hits.get(CATEGORY_NAMES[order[0]], []),
        }

    def predict(self, title: str = "", description: str = "", advertiser: str = "", domain: str = "") -> dict:
        text = build_text(title, description, advertiser, domain)
        if not text:
            return {"category": "Sponsored Stories", "confidence": 0.0, "method": "empty",
                    "needs_review": True, "alternatives": [], "matched_keywords": []}
        lex = self._proba(self.lexical, [text])[0]
        sem = self._proba(self.semantic, _Encoder.encode([text]))[0] if self.semantic is not None else None
        return self._combine(text, lex, sem, advertiser_key(advertiser, domain))

    def predict_many(self, ads: list[dict]) -> list[dict]:
        """Batch version (one encoder call) used when storing scraped ads."""
        if not ads:
            return []
        texts = [build_text(a.get("title", ""), a.get("description", ""), a.get("advertiser", ""), a.get("domain", ""))
                 for a in ads]
        lex = self._proba(self.lexical, texts)
        sem = self._proba(self.semantic, _Encoder.encode(texts)) if self.semantic is not None else None
        return [self._combine(t, lex[i], sem[i] if sem is not None else None,
                              advertiser_key(a.get("advertiser", ""), a.get("domain", "")))
                for i, (t, a) in enumerate(zip(texts, ads))]


def category_colors() -> dict[str, str]:
    return {name: color for name, (color, _) in CATEGORIES.items()}
