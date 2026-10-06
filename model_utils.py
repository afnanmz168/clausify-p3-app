"""
Inference logic for the CUAD Contract Analyzer app.

Loads the two trained transformer models and runs the exact inference
path used in the training notebook:

  Model 1  (Presence)  DistilBertForSequenceClassification
                       (question, window) -> does this window contain the clause?
                       max-pool over all windows of a contract -> present / absent.

  Model 2  (Span)      DistilBertForQuestionAnswering
                       (question, window) -> start/end token of the clause text.

Both run on CPU for portability (no GPU / MPS memory pressure).
"""

import os
import re
import json
import functools

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
)

# --------------------------------------------------------------------------- #
# Paths. Where the trained models are found, in order:
#   1. the CUAD_MODELS_DIR environment variable;
#   2. a "models/" folder next to this file;
#   3. the sibling "final project p3/notebooks/outputs" training folder (local use);
#   4. otherwise they are downloaded once from the Hugging Face Hub repo in
#      CLAUSIFY_HF_REPO (default af123Af/clausify-models) — this is what a cloud
#      deployment such as Streamlit Community Cloud uses.
# --------------------------------------------------------------------------- #
BASE = os.path.dirname(os.path.abspath(__file__))
HF_REPO = os.environ.get("CLAUSIFY_HF_REPO", "af123Af/clausify-models")


def _has_models(d):
    return all(os.path.exists(os.path.join(d, m, "final", "config.json"))
               for m in ("presence_mil", "span"))


def _resolve_models_dir():
    for d in (os.environ.get("CUAD_MODELS_DIR"),
              os.path.join(BASE, "models"),
              os.path.join(os.path.dirname(BASE), "final project p3", "notebooks", "outputs")):
        if d and _has_models(d):
            return d
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(HF_REPO, allow_patterns=["presence_mil/*", "span/*", "summarizer/*", "baseline/*"])
    except Exception:            # offline / repo missing: app.py shows a clear error
        return os.environ.get("CUAD_MODELS_DIR", os.path.join(BASE, "models"))


MODELS_DIR = _resolve_models_dir()
PRESENCE_DIR = os.path.join(MODELS_DIR, "presence_mil", "final")
SPAN_DIR = os.path.join(MODELS_DIR, "span", "final")
SUMMARIZER_DIR = os.path.join(MODELS_DIR, "summarizer", "final")
# TF-IDF baseline: "baseline/baseline.pkl" in the Hub repo, or the training artifacts folder locally.
BASELINE_PATH = next((p for p in (os.path.join(MODELS_DIR, "baseline", "baseline.pkl"),
                                  os.path.join(os.path.dirname(MODELS_DIR), "artifacts", "baseline.pkl"))
                      if os.path.exists(p)), os.path.join(MODELS_DIR, "baseline", "baseline.pkl"))

DEVICE = "cpu"

# Windowing — identical to the notebook (2000-char windows, 1500 stride).
WIN, STRIDE = 2000, 1500
MAX_WINDOWS = 30          # cap for very long contracts (keeps the demo snappy)
PRESENCE_MAXLEN = 256
SPAN_MAXLEN = 320
FOCUS = 1200              # span focus sub-window

# 41 category -> CUAD question text (the model was trained on these exact strings)
with open(os.path.join(BASE, "cat_questions.json")) as f:
    CAT_QUESTIONS = json.load(f)
CATEGORIES = list(CAT_QUESTIONS.keys())

# --------------------------------------------------------------------------- #
# Risk taxonomy — from the poster: 8 High, 22 Medium, 11 Low (41 total),
# scored from the *weaker party's* point of view. Each entry: (level, reason).
# --------------------------------------------------------------------------- #
RISK = {
    # -------- HIGH (8) --------
    "Non-Compete":                       ("High",   "Limits your future income and freedom to work."),
    "Uncapped Liability":                ("High",   "Unlimited damages — no ceiling on what you can owe."),
    "Ip Ownership Assignment":           ("High",   "You lose ownership of a core asset for good."),
    "Exclusivity":                       ("High",   "Locks you to a single source or buyer."),
    "Most Favored Nation":               ("High",   "You must always give the other side your best terms."),
    "Liquidated Damages":                ("High",   "Fixed penalties trigger automatically on breach."),
    "Irrevocable Or Perpetual License":  ("High",   "A grant you can never take back."),
    "Minimum Commitment":                ("High",   "You are forced to buy or deliver a minimum amount."),

    # -------- MEDIUM (22) --------
    "Renewal Term":                      ("Medium", "Auto-renewal can extend the contract unexpectedly."),
    "Notice Period To Terminate Renewal":("Medium", "Miss the window and you are locked in for another term."),
    "Competitive Restriction Exception": ("Medium", "Carve-outs to competitive limits — read carefully."),
    "No-Solicit Of Customers":           ("Medium", "Restricts approaching the other side's customers."),
    "No-Solicit Of Employees":           ("Medium", "Restricts hiring the other side's staff."),
    "Non-Disparagement":                 ("Medium", "Limits what you can publicly say."),
    "Termination For Convenience":       ("Medium", "The other side can walk away with notice."),
    "Rofr/Rofo/Rofn":                    ("Medium", "First-refusal rights can constrain your options."),
    "Change Of Control":                 ("Medium", "A sale or merger may trigger termination."),
    "Anti-Assignment":                   ("Medium", "You cannot transfer the contract without consent."),
    "Revenue/Profit Sharing":            ("Medium", "You must share a slice of revenue or profit."),
    "Price Restrictions":                ("Medium", "Limits what prices you may set or change."),
    "Volume Restriction":                ("Medium", "Caps how much you can sell or distribute."),
    "Joint Ip Ownership":                ("Medium", "IP is co-owned — control is shared, not sole."),
    "Non-Transferable License":          ("Medium", "The licence cannot be passed on."),
    "Affiliate License-Licensor":        ("Medium", "Licensor's affiliates are pulled into the grant."),
    "Unlimited/All-You-Can-Eat-License": ("Medium", "Broad, unlimited licence scope — check who benefits."),
    "Source Code Escrow":                ("Medium", "Source is held by a third party under conditions."),
    "Post-Termination Services":         ("Medium", "Obligations continue after the contract ends."),
    "Audit Rights":                      ("Medium", "The other side can inspect your records."),
    "Cap On Liability":                  ("Medium", "Caps recovery — may under-compensate real losses."),
    "Covenant Not To Sue":               ("Medium", "You waive the right to sue over listed matters."),

    # -------- LOW (11) --------
    "Document Name":                     ("Low",    "Identifies the agreement — informational."),
    "Parties":                           ("Low",    "Names who is signing — informational."),
    "Agreement Date":                    ("Low",    "The signing date — informational."),
    "Effective Date":                    ("Low",    "When terms start — informational."),
    "Expiration Date":                   ("Low",    "When the contract ends — informational."),
    "Governing Law":                     ("Low",    "Which jurisdiction's law applies — neutral."),
    "License Grant":                     ("Low",    "You receive usage rights — generally favorable."),
    "Warranty Duration":                 ("Low",    "How long the warranty protects you."),
    "Insurance":                         ("Low",    "Required coverage — a protection."),
    "Affiliate License-Licensee":        ("Low",    "Extends rights to your affiliates — favorable."),
    "Third Party Beneficiary":           ("Low",    "Names an outside beneficiary — informational."),
}

RISK_ORDER = {"High": 0, "Medium": 1, "Low": 2}


def risk_of(category):
    """Return (level, reason) for a category; defaults to Medium if unknown."""
    return RISK.get(category, ("Medium", ""))


def models_available():
    """True if both trained models are on disk."""
    return (
        os.path.exists(os.path.join(PRESENCE_DIR, "config.json"))
        and os.path.exists(os.path.join(SPAN_DIR, "config.json"))
    )


@functools.lru_cache(maxsize=1)
def _load_presence():
    tok = AutoTokenizer.from_pretrained(PRESENCE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(PRESENCE_DIR)
    model.to(DEVICE).eval()
    return tok, model


@functools.lru_cache(maxsize=1)
def _load_span():
    tok = AutoTokenizer.from_pretrained(SPAN_DIR)
    model = AutoModelForQuestionAnswering.from_pretrained(SPAN_DIR)
    model.to(DEVICE).eval()
    return tok, model


def warm_up():
    """Load both models so the first analysis isn't slow."""
    _load_presence()
    _load_span()
    if baseline_available():
        _load_baseline()
    if summarizer_available():
        _load_summarizer()


def make_windows(text):
    """Slide 2000-char windows with 1500 stride over the contract."""
    wins, pos = [], 0
    while pos < len(text):
        wins.append(text[pos:pos + WIN])
        if pos + WIN >= len(text):
            break
        pos += STRIDE
    wins = wins or [text]
    return wins[:MAX_WINDOWS]


def predict_presence(text, progress=None):
    """
    For every category, max-pool the window-level classifier over all windows.

    Returns:
        scores       {category: max probability the clause is present}
        best_window  {category: the window that scored highest (for span)}
    """
    tok, model = _load_presence()
    wins = make_windows(text)
    scores, best_window = {}, {}
    n = len(CATEGORIES)

    for ci, cat in enumerate(CATEGORIES):
        q = CAT_QUESTIONS[cat]
        win_scores = []
        for i in range(0, len(wins), 16):
            batch = wins[i:i + 16]
            enc = tok(
                [q] * len(batch), batch,
                truncation=True, max_length=PRESENCE_MAXLEN,
                padding=True, return_tensors="pt",
            ).to(DEVICE)
            with torch.no_grad():
                logits = model(**enc).logits
            p1 = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            win_scores.extend(p1.tolist())
        win_scores = np.array(win_scores)
        scores[cat] = float(win_scores.max())
        best_window[cat] = wins[int(win_scores.argmax())]
        if progress is not None:
            progress((ci + 1) / n)

    return scores, best_window


def extract_span(question, window):
    """
    Run the QA model over the best window (in FOCUS-char sub-chunks) and
    return the highest-confidence span.

    Returns: (span_text, confidence)
    """
    tok, model = _load_span()
    best_text, best_conf = "", -1.0

    starts = list(range(0, max(1, len(window) - 300), FOCUS - 300)) or [0]
    for start in starts:
        ctx = window[start:start + FOCUS]
        enc = tok(
            question, ctx,
            truncation="only_second", max_length=SPAN_MAXLEN,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            out = model(**enc)

        ids = enc["input_ids"][0]
        sep_positions = (ids == tok.sep_token_id).nonzero()
        sep = int(sep_positions[0].item())          # end of the question
        sl = out.start_logits[0].cpu().numpy()
        el = out.end_logits[0].cpu().numpy()

        mask = np.full(len(ids), -1e9)
        mask[sep + 1:] = 0.0                         # only look inside the context
        s = int((sl + mask).argmax())
        e = int((el + mask).argmax())
        if e < s:
            e = s
        e = min(e, s + 80)                           # cap span length

        span = tok.decode(ids[s:e + 1], skip_special_tokens=True).strip()
        conf = float(
            F.softmax(out.start_logits, dim=-1)[0][s]
            * F.softmax(out.end_logits, dim=-1)[0][e]
        )
        if span and conf > best_conf:
            best_text, best_conf = span, conf

    return best_text, best_conf


def _paragraphs(text, max_len=700):
    """Split a window into paragraphs; long paragraphs are split into sentences."""
    out = []
    for p in re.split(r"\n\s*\n", text):
        p = p.strip()
        if len(p) > max_len:
            out.extend(s.strip() for s in re.split(r"(?<=[.;:])\s+(?=[A-Z(\"“0-9])", p))
        elif p:
            out.append(p)
    out = [p for p in out if len(p) >= 30]
    return out or [text.strip()]


def locate_clause(category, window):
    """
    Pick the paragraph of `window` that the presence model scores highest for
    `category`. The span model alone is not a reliable locator: it was trained on
    focus windows with the answer ~150 chars in and tends to return whatever text
    sits there, whatever the question. The presence model does read the question,
    so it chooses the paragraph and the span model only highlights inside it.

    Returns: (paragraph, highlighted_span)
    """
    tok, model = _load_presence()
    paras = _paragraphs(window)
    q = CAT_QUESTIONS[category]
    enc = tok([q] * len(paras), paras, truncation=True, max_length=PRESENCE_MAXLEN,
              padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        lg = model(**enc).logits
    para = paras[int((lg[:, 1] - lg[:, 0]).argmax())]
    span, _ = extract_span(q, para)
    return para, span if span and span.lower() in para.lower() else ""


RULES = {
    "AND": "Balanced — AND-ensemble (TF-IDF and DistilBERT must agree)",
    "TRANS": "Cautious — DistilBERT alone (catches more High-risk clauses)",
}


def _full_paragraph(text, para, max_len=1500):
    """A located paragraph can start mid-word when a 2,000-char window cut it; extend it to
    the whole paragraph of the original contract (if that stays reasonably short)."""
    start = text.find(para)
    if start < 0:
        return para, -1
    s = text.rfind("\n\n", 0, start)
    s = 0 if s < 0 else s + 2
    e = text.find("\n\n", start + len(para))
    e = len(text) if e < 0 else e
    full = text[s:e].strip()
    if len(full) > max_len or not full:
        return para, start
    return full, text.find(full, s)


def analyze(text, threshold=0.5, rule="TRANS", progress=None):
    """
    Full pipeline for one contract.

    rule "AND"  : a category is present when min(TF-IDF, DistilBERT) >= threshold
                  (the report's headline configuration, micro-F1 0.779);
    rule "TRANS": DistilBERT max-pooled score alone (higher High-risk recall).
    Falls back to "TRANS" if the TF-IDF baseline is not available.

    Returns (present, scores):
        present  list of dicts, one per present category, sorted High -> Low then by score:
                 {category, score, transformer_score, tfidf_score, span_text, highlight,
                  start, risk, reason}
        scores   {category: {"transformer": p, "tfidf": p or None, "score": p}}
    """
    trans, best_window = predict_presence(text, progress=progress)
    tfidf = tfidf_scores(text) if baseline_available() else {}
    if rule == "AND" and not tfidf:
        rule = "TRANS"

    scores, present = {}, []
    for cat, pt in trans.items():
        pf = tfidf.get(cat)
        sc = min(pt, pf) if rule == "AND" else pt
        scores[cat] = {"transformer": pt, "tfidf": pf, "score": sc}
        if sc >= threshold:
            para, span = locate_clause(cat, best_window[cat])
            para, start = _full_paragraph(text, para)
            level, reason = risk_of(cat)
            present.append({
                "category": cat, "score": sc, "transformer_score": pt, "tfidf_score": pf,
                "span_text": para,          # the located paragraph, quoted on the card
                "highlight": span,          # span-model highlight inside it ("" if none)
                "start": start,             # position in the contract (-1 if not found)
                "risk": level, "reason": reason,
            })

    # sort by risk (High → Low), then by score within a risk band
    present.sort(key=lambda d: (RISK_ORDER[d["risk"]], -d["score"]))
    return present, scores


# --------------------------------------------------------------------------- #
# TF-IDF + logistic-regression baseline (document level, one classifier per
# category) — the lexical half of the AND-ensemble.
# --------------------------------------------------------------------------- #
def baseline_available():
    return os.path.exists(BASELINE_PATH)


@functools.lru_cache(maxsize=1)
def _load_baseline():
    import pickle
    with open(BASELINE_PATH, "rb") as f:
        return pickle.load(f)


def tfidf_scores(text):
    """{category: probability the category is present} from the full-document TF-IDF model."""
    bl = _load_baseline()
    X = bl["vec"].transform([text])
    out = {}
    for cat in CATEGORIES:
        kind, obj = bl["classifiers"][cat]
        out[cat] = float(obj) if kind == "const" else float(obj.predict_proba(X)[0, 1])
    return out


# --------------------------------------------------------------------------- #
# Summarizer — fine-tuned FLAN-T5-small. The report (Section 5.1) shows it maps
# a clause to one of 41 category-level plain-English sentences rather than
# summarizing the specific wording; the app labels its output accordingly.
# --------------------------------------------------------------------------- #
def summarizer_available():
    return os.path.exists(os.path.join(SUMMARIZER_DIR, "config.json"))


@functools.lru_cache(maxsize=1)
def _load_summarizer():
    from transformers import AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained(SUMMARIZER_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_DIR).to(DEVICE).eval()
    return tok, model


_TEMPLATES_PATH = os.path.join(BASE, "summarizer_templates.json")


@functools.lru_cache(maxsize=1)
def _template_to_category():
    """The summarizer was trained on 41 category sentences; map each back to its category."""
    try:
        t = json.load(open(_TEMPLATES_PATH))
    except OSError:
        return {}
    return {_norm_sentence(v): k for k, v in t.items()}


def _norm_sentence(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def summary_category(sentence):
    """Category whose training sentence the summarizer reproduced, or None if it wrote something else."""
    return _template_to_category().get(_norm_sentence(sentence))


def summarize(texts, progress=None, batch=8):
    """One plain-English sentence per input clause (same prompt and decoding as the test)."""
    tok, model = _load_summarizer()
    out = []
    for i in range(0, len(texts), batch):
        chunk = [f"summarize in plain English: {t[:1200]}" for t in texts[i:i + batch]]
        enc = tok(chunk, return_tensors="pt", truncation=True, max_length=256, padding=True).to(DEVICE)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=48, num_beams=2)
        out.extend(tok.decode(g, skip_special_tokens=True).strip() for g in gen)
        if progress is not None:
            progress(min(1.0, (i + batch) / len(texts)))
    return out


# --------------------------------------------------------------------------- #
# Missing-protection checklist. What a contract does NOT contain can matter to
# the weaker party as much as what it does. Each entry: category, why its
# absence matters, and (optionally) a category that must be present for the
# check to apply. Absence means "not detected" — the models can miss clauses.
# --------------------------------------------------------------------------- #
PROTECTIONS = [
    ("Cap On Liability", "Without a liability cap, the amount you could owe for a breach may have no upper limit.", None),
    ("Termination For Convenience", "Without it, you may have no way to exit the contract early if it stops working for you.", None),
    ("Governing Law", "Without a governing-law clause, it is unclear whose law decides a dispute, which can make one costly.", None),
    ("Expiration Date", "No clear end date — check how long the obligations actually last.", None),
    ("Warranty Duration", "No stated warranty period — it may be unclear how long you are protected against defects.", None),
    ("Insurance", "No insurance requirement — losses may not be covered by anyone's policy.", None),
    ("Notice Period To Terminate Renewal", "The contract renews, but no notice period to stop the renewal was detected — you may be locked in.", "Renewal Term"),
]


def missing_protections(found_categories):
    found = set(found_categories)
    return [{"category": c, "why": why, "risk_if_missing": "check"}
            for c, why, needs in PROTECTIONS
            if c not in found and (needs is None or needs in found)]

# --------------------------------------------------------------------------- #
# Clause mode — the user pastes a list of separate clauses and expects one
# categorized card per clause.
#
# The presence model scores (category question, text). On a single clause some
# categories score high on almost anything ("Parties" fires on any text naming
# the two sides), so each category's log-odds is standardized against how it
# scores on real CUAD clauses of OTHER categories (clause_calibration.json, from
# the training split; see calibrate_clause_mode.py), and the clause gets the
# category with the highest standardized score.
# --------------------------------------------------------------------------- #
_CAL_PATH = os.path.join(BASE, "clause_calibration.json")
MIN_CLAUSE_CHARS = 40          # shorter fragments (titles, signature lines) are skipped
AUTO_MAX_CLAUSES = 25          # auto mode: at most this many paragraphs ...
AUTO_MAX_CHARS = 12000         # ... and this much text to be treated as a clause list


@functools.lru_cache(maxsize=1)
def _calibration():
    cal = json.load(open(_CAL_PATH))
    assert cal["categories"] == CATEGORIES, "calibration was built for a different category order"
    return np.array(cal["neg_mean"]), np.array(cal["neg_std"]), float(cal["unrecognized_below_z"])


def clause_eval():
    """Held-out accuracy of the model-only clause classifier (from the calibration run)."""
    try:
        return json.load(open(_CAL_PATH))["eval"]
    except (OSError, KeyError, ValueError):
        return None


def clause_mode_available():
    return os.path.exists(_CAL_PATH)


def split_clauses(text):
    """Split pasted text into clauses: blank-line separated paragraphs, or, if the
    text has no blank lines, lines that start a numbered item (1. / 2) / (a) ...)."""
    text = text.replace("\r\n", "\n").strip()
    parts = [p.strip() for p in re.split(r"\n\s*\n", text)]
    if len(parts) == 1:
        parts = [p.strip() for p in
                 re.split(r"\n(?=\s*(?:\d+[\.\)]|\([a-z0-9]+\)|[a-z][\.\)])\s)", text)]
    return [p for p in parts if len(p) >= MIN_CLAUSE_CHARS]


def looks_like_clause_list(text):
    clauses = split_clauses(text)
    return 2 <= len(clauses) <= AUTO_MAX_CLAUSES and len(text) <= AUTO_MAX_CHARS


# Clause headings ("2. EXCLUSIVITY. ...") that name a category are stronger evidence
# than the model. Only headings that unambiguously name one category are used;
# anything else (e.g. "Limitation of Liability", which may hide an UNcapped
# liability) is left to the model.
_HEADING = re.compile(
    r"^\s*(?:(?:section|article|clause)\s+)?(?:[0-9ivxIVX]+(?:\.[0-9]+)*[\.\)]?|\([a-z0-9]+\))?\s*"
    r"([A-Za-z][A-Za-z&/\-’' ]{2,60}?)\s*[\.:—–-]\s", re.IGNORECASE)
_HEADING_SYNONYMS = {
    "intellectual property assignment": "Ip Ownership Assignment",
    "assignment of intellectual property": "Ip Ownership Assignment",
    "ip assignment": "Ip Ownership Assignment",
    "ownership of intellectual property": "Ip Ownership Assignment",
    "perpetual license": "Irrevocable Or Perpetual License",
    "irrevocable license": "Irrevocable Or Perpetual License",
    "most favoured nation": "Most Favored Nation",
    "non-competition": "Non-Compete", "noncompete": "Non-Compete", "non compete": "Non-Compete",
    "term and renewal": "Renewal Term", "renewal": "Renewal Term", "automatic renewal": "Renewal Term",
    "expiration": "Expiration Date", "assignment": "Anti-Assignment", "audit": "Audit Rights",
    "minimum purchase": "Minimum Commitment", "minimum purchase commitment": "Minimum Commitment",
    "grant of license": "License Grant", "license": "License Grant", "licence grant": "License Grant",
    "choice of law": "Governing Law", "applicable law": "Governing Law",
    "source code escrow agreement": "Source Code Escrow", "escrow": "Source Code Escrow",
    "non-disparagement": "Non-Disparagement", "warranty period": "Warranty Duration",
}


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9/& \-]", " ", s.lower())).strip()


_HEADING_TO_CAT = {_norm(c): c for c in CATEGORIES}
_HEADING_TO_CAT.update({_norm(k): v for k, v in _HEADING_SYNONYMS.items()})


def heading_category(clause):
    """Category named by the clause's leading heading, or None."""
    m = _HEADING.match(clause)
    if not m or len(m.group(1).split()) > 6:
        return None
    return _HEADING_TO_CAT.get(_norm(m.group(1)))


def classify_clauses(clauses, progress=None):
    """
    One card per clause. Returns a list of dicts in input order:
        {index, clause, category, risk, reason, source, model_guess, z, presence_score, best_guess, runner_up}
    source is "heading" when the clause heading names the category, else "model".
    category is None (risk "Unrecognized") when no category stands out.
    """
    tok, model = _load_presence()
    mean, std, z_cut = _calibration()
    qs = [CAT_QUESTIONS[c] for c in CATEGORIES]
    out = []
    for i, cl in enumerate(clauses):
        enc = tok(qs, [cl] * len(qs), truncation=True, max_length=PRESENCE_MAXLEN,
                  padding=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            lg = model(**enc).logits
        log_odds = (lg[:, 1] - lg[:, 0]).cpu().numpy()
        prob = F.softmax(lg, dim=-1)[:, 1].cpu().numpy()
        z = (log_odds - mean) / std
        order = np.argsort(-z)
        best, second = int(order[0]), int(order[1])
        by_heading = heading_category(cl)
        if by_heading:
            cat, source = by_heading, "heading"
        else:
            cat, source = (CATEGORIES[best], "model") if z[best] >= z_cut else (None, "model")
        level, reason = risk_of(cat) if cat else ("Unrecognized", "No clause category stands out clearly — review manually.")
        out.append({
            "index": i + 1, "clause": cl, "category": cat, "risk": level, "reason": reason,
            "source": source, "model_guess": CATEGORIES[best],
            "z": float(z[best]), "presence_score": float(prob[best]),
            "best_guess": CATEGORIES[best], "runner_up": CATEGORIES[second],
        })
        if progress is not None:
            progress((i + 1) / len(clauses))
    return out
