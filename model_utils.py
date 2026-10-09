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
        return snapshot_download(HF_REPO, allow_patterns=["presence_mil/*", "presence_v2/*", "span/*", "span_v2/*", "clause_v2/*",
                                                          "summarizer/*", "baseline/*", "baseline_v2/*"])
    except Exception:            # offline / repo missing: app.py shows a clear error
        return os.environ.get("CUAD_MODELS_DIR", os.path.join(BASE, "models"))


MODELS_DIR = _resolve_models_dir()
PRESENCE_DIR = os.path.join(MODELS_DIR, "presence_mil", "final")
SPAN_DIR = os.path.join(MODELS_DIR, "span", "final")
# Retrained span model (report Section 5.2.1): trained on the same 1,200-character chunks it reads at
# run time, with the answer anywhere in the chunk or absent. span_v2.json holds the decoding and the
# way the quote is chosen, both picked on the 81 validation contracts. Without it, the first model.
SPAN_V2_DIR = os.path.join(MODELS_DIR, "span_v2", "final")
_SPAN_V2_CFG = os.path.join(BASE, "span_v2.json")
SPAN_V2 = json.load(open(_SPAN_V2_CFG)) if os.path.exists(_SPAN_V2_CFG) else None
SUMMARIZER_DIR = os.path.join(MODELS_DIR, "summarizer", "final")
# TF-IDF baseline: "baseline/baseline.pkl" in the Hub repo, or the training artifacts folder locally.
BASELINE_PATH = next((p for p in (os.path.join(MODELS_DIR, "baseline", "baseline.pkl"),
                                  os.path.join(os.path.dirname(MODELS_DIR), "artifacts", "baseline.pkl"))
                      if os.path.exists(p)), os.path.join(MODELS_DIR, "baseline", "baseline.pkl"))

# Version 2 (October 2026 re-run, report Section 5.3.3): the presence model reads the whole
# 2,000-character window (512 tokens), and the TF-IDF model, the decision rule and one threshold
# per category were all chosen on a validation split. decision_v2.json holds those choices.
PRESENCE_V2_DIR = os.path.join(MODELS_DIR, "presence_v2", "final")
BASELINE_V2_PATH = next((p for p in (os.path.join(MODELS_DIR, "baseline_v2", "baseline.pkl"),
                                     os.path.join(os.path.dirname(MODELS_DIR), "artifacts", "baseline_v2.pkl"))
                         if os.path.exists(p)), os.path.join(MODELS_DIR, "baseline_v2", "baseline.pkl"))
with open(os.path.join(BASE, "decision_v2.json")) as f:
    DECISION_V2 = json.load(f)
# Isotonic map from the re-run transformer's max-pooled score to a calibrated chance, fitted on the
# 81 validation contracts (test-set ECE 0.206 -> 0.016). Written by downstream_v2.py.
with open(os.path.join(BASE, "calibration_v2.json")) as f:
    _CALIB_V2 = json.load(f)


# A card whose calibrated chance is below this is labelled "Possible — check" and shown apart from the
# others. The recall-first thresholds are set per clause type, the calibration map is shared, so some
# types are reported at a chance well under 50%: on the 102 test contracts 728 of the default setting's
# 1,963 cards (37%) fall below it. Only 30% of those are real clauses, but they include 46 of the 159
# High-risk clauses found, so they are flagged rather than dropped.
POSSIBLE_BELOW = 0.5


def calibrated(p):
    """Calibrated chance that a clause type is present, from the re-run transformer's score.
    Shown between 1% and 99%: no single prediction is certain, whatever the map says."""
    return float(min(0.99, max(0.01, np.interp(p, _CALIB_V2["x"], _CALIB_V2["y"]))))

DEVICE = "cpu"

# Windowing — identical to the notebook (2000-char windows, 1500 stride).
WIN, STRIDE = 2000, 1500
MAX_WINDOWS = 30          # every clause type reads the first 30 windows (keeps the app fast)
# Version 2: in longer contracts, the TF-IDF model picks this many of the later windows for each clause
# type, and the presence model reads those too. Chosen on the 81 validation contracts (smallest number
# whose micro-F2 is within 0.005 of reading every window); on the test contracts the default setting
# then finds 88.6% of High-risk clauses, against 77.3% with the first 30 windows alone and 90.3% with
# every window (final project p3/retrain/v2_fullwindow/window_cap.py).
EXTRA_WINDOWS = 5
PRESENCE_MAXLEN = 256
PRESENCE_V2_MAXLEN = DECISION_V2["max_len"]
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


def v2_available():
    """True if the version-2 presence model and its TF-IDF model are on disk."""
    return os.path.exists(os.path.join(PRESENCE_V2_DIR, "config.json")) and os.path.exists(BASELINE_V2_PATH)


@functools.lru_cache(maxsize=2)
def _load_presence(version="v1"):
    d = PRESENCE_V2_DIR if version == "v2" else PRESENCE_DIR
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForSequenceClassification.from_pretrained(d)
    model.to(DEVICE).eval()
    return tok, model


def _presence_maxlen(version):
    return PRESENCE_V2_MAXLEN if version == "v2" else PRESENCE_MAXLEN


def span_v2_available():
    return SPAN_V2 is not None and os.path.exists(os.path.join(SPAN_V2_DIR, "config.json"))


@functools.lru_cache(maxsize=2)
def _load_span(version="v1"):
    d = SPAN_V2_DIR if version == "v2" else SPAN_DIR
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForQuestionAnswering.from_pretrained(d)
    model.to(DEVICE).eval()
    return tok, model


def warm_up():
    """Load the models used by the default settings so the first analysis isn't slow."""
    if v2_available():
        _load_presence("v2")
        _load_baseline("v2")
    else:
        _load_presence()
        if baseline_available():
            _load_baseline()
    _load_span("v2" if span_v2_available() else "v1")
    if summarizer_available():
        _load_summarizer()


def all_windows(text):
    """Slide 2000-char windows with 1500 stride over the whole contract."""
    wins, pos = [], 0
    while pos < len(text):
        wins.append(text[pos:pos + WIN])
        if pos + WIN >= len(text):
            break
        pos += STRIDE
    return wins or [text]


def make_windows(text):
    """The first MAX_WINDOWS windows, which every clause type reads."""
    return all_windows(text)[:MAX_WINDOWS]


def _extra_windows(later, version):
    """{category: indices into `later`}: the EXTRA_WINDOWS windows after the first MAX_WINDOWS that the
    re-run TF-IDF model, applied to each window alone, rates most likely to contain that category.
    Ties (and the constant categories) keep document order, exactly as in window_cap.py."""
    if version != "v2" or not later or EXTRA_WINDOWS <= 0:
        return {}
    bl = _load_baseline("v2")
    X = bl["vec"].transform(later)
    out = {}
    for cat in CATEGORIES:
        kind, obj = bl["classifiers"][cat]
        s = np.zeros(len(later)) if kind == "const" else obj.predict_proba(X)[:, 1]
        out[cat] = [int(i) for i in np.argsort(-s, kind="stable")[:EXTRA_WINDOWS]]
    return out


def predict_presence(text, progress=None, version="v1"):
    """
    For every category, max-pool the window-level classifier over the windows it reads: the first
    MAX_WINDOWS, plus (version 2, long contracts) the EXTRA_WINDOWS later ones the TF-IDF model picks.

    Returns:
        scores       {category: max probability the clause is present}
        best_window  {category: the window that scored highest (for span)}
    """
    tok, model = _load_presence(version)
    maxlen = _presence_maxlen(version)
    allw = all_windows(text)
    first, later = allw[:MAX_WINDOWS], allw[MAX_WINDOWS:]
    extra = _extra_windows(later, version)
    scores, best_window = {}, {}
    n = len(CATEGORIES)

    for ci, cat in enumerate(CATEGORIES):
        q = CAT_QUESTIONS[cat]
        wins = first + [later[i] for i in extra.get(cat, [])]
        win_scores = []
        for i in range(0, len(wins), 16):
            batch = wins[i:i + 16]
            enc = tok(
                [q] * len(batch), batch,
                truncation=True, max_length=maxlen,
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
    tok, model = _load_span("v1")
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


def _chunk_starts(n, size, stride):
    s = list(range(0, max(1, n - size + 1), stride))
    if s[-1] + size < n:
        s.append(n - size)
    return s


def extract_span_v2(question, text):
    """
    Retrained span model, read exactly as in training and in the report's test: chunks of
    span_v2.json["chunk"] characters every ["stride"], the answer scored as start + end logit
    (minus the chunk's "no answer" score if ["rule"] is "minus_null"), at most ["max_answer_tokens"].

    Returns: (answer text as written in `text`, start offset, end offset, score); ("", -1, -1, -inf)
    for empty text.
    """
    cfg = SPAN_V2
    tok, model = _load_span("v2")
    if not text.strip():
        return "", -1, -1, float("-inf")
    cs = _chunk_starts(len(text), cfg["chunk"], cfg["stride"])
    chunks = [text[a:a + cfg["chunk"]] for a in cs]
    enc = tok([question] * len(chunks), chunks, truncation="only_second", max_length=cfg["max_len"],
              padding=True, return_offsets_mapping=True, return_tensors="pt")
    off = enc.pop("offset_mapping").numpy()
    with torch.no_grad():
        out = model(**enc.to(DEVICE))
    sl, el = out.start_logits.cpu().numpy(), out.end_logits.cpu().numpy()
    best = ("", -1, -1, float("-inf"))
    for k in range(len(chunks)):
        ctx = np.array([x == 1 for x in enc.sequence_ids(k)])
        s = int(np.argmax(np.where(ctx, sl[k], -1e9)))
        pos = np.arange(len(ctx))
        e = int(np.argmax(np.where(ctx & (pos >= s) & (pos <= s + cfg["max_answer_tokens"]), el[k], -1e18)))
        score = float(sl[k][s] + el[k][e])
        if cfg["rule"] == "minus_null":
            score -= float(sl[k][0] + el[k][0])
        if score > best[3]:
            a, b = int(off[k][s][0]) + cs[k], int(off[k][e][1]) + cs[k]
            best = (text[a:b].strip(), a, b, score)
    return best


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


def locate_clause(category, window, version="v1"):
    """
    Pick the paragraph of `window` that the presence model scores highest for
    `category`. The span model alone is not a reliable locator: it was trained on
    focus windows with the answer ~150 chars in and tends to return whatever text
    sits there, whatever the question. The presence model does read the question,
    so it chooses the paragraph and the span model only highlights inside it.

    Returns: (paragraph, highlighted_span)
    """
    q = CAT_QUESTIONS[category]
    paras = _paragraphs(window)
    if span_v2_available() and SPAN_V2["quote"] == "span_paragraph":
        # the retrained span model finds the clause in the window; quote the paragraph around it
        span, a, _, _ = extract_span_v2(q, window)
        pos = [window.find(p) for p in paras]
        hit = [p for p, st in zip(paras, pos) if 0 <= st <= a < st + len(p)]
        para = hit[0] if hit else min(zip(paras, pos), key=lambda x: abs(x[1] - a))[0]
        return para, span if span and span in para else ""
    tok, model = _load_presence(version)
    enc = tok([q] * len(paras), paras, truncation=True, max_length=_presence_maxlen(version),
              padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        lg = model(**enc).logits
    para = paras[int((lg[:, 1] - lg[:, 0]).argmax())]
    if span_v2_available():
        span = extract_span_v2(q, para)[0]
        return para, span if span and span in para else ""
    span, _ = extract_span(q, para)
    return para, span if span and span.lower() in para.lower() else ""


RULES = {
    # version 2: settings chosen on a validation split, tested once (report Section 5.3.3)
    "RECALL": "Recall-first — DistilBERT (misses the fewest High-risk clauses)",
    "RECALL_ENS": "Recall-first ensemble — DistilBERT and TF-IDF averaged (for long contracts)",
    "BALANCED": "Balanced — both models must agree (highest overall F1, misses more High-risk clauses)",
    # version 1 fallback, used only when the version-2 files are not available
    "TRANS": "Original model — DistilBERT alone",
    "AND": "Original model — AND-ensemble (TF-IDF and DistilBERT must agree)",
}
# rule -> (tuning objective in decision_v2.json, which models decide). The TF-IDF model reads
# whole-document word statistics learnt from long SEC filings and scores short contracts low,
# so the default uses the DistilBERT half alone (report Section 5.3.3).
V2_RULES = {"RECALL": ("f2", "transformer"), "RECALL_ENS": ("f2", "ensemble"), "BALANCED": ("f1", "ensemble")}


def default_rule():
    return "RECALL" if v2_available() else "TRANS"


def rule_options():
    if v2_available():
        return ["RECALL", "RECALL_ENS", "BALANCED"]
    return ["TRANS", "AND"] if baseline_available() else ["TRANS"]


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


def _decide_v2(rule, cat, pt, pf):
    """(present?, score shown, threshold shown) under the tested version-2 decision rule."""
    objective, deciders = V2_RULES[rule]
    d = DECISION_V2["objectives"][objective]
    if deciders == "transformer":
        thr = d["t_transformer"][cat]
        return pt >= thr, pt, thr
    if d["rule"] == "AVG":
        sc, thr = (pt + pf) / 2, d["t_avg"][cat]
        return sc >= thr, sc, thr
    ok_t, ok_f = pt >= d["t_transformer"][cat], pf >= d["t_tfidf"][cat]
    hit = (ok_t and ok_f) if d["rule"] == "AND" else (ok_t or ok_f)
    return hit, min(pt, pf), None


def analyze(text, threshold=0.5, rule=None, progress=None):
    """
    Full pipeline for one contract.

    Version 2 (default when its files are present; report Section 5.3.3), with the presence model
    reading the whole window and every setting chosen on a validation split:
      rule "RECALL"    : DistilBERT score >= its per-category threshold, chosen to maximise
                         micro-F2 (a missed clause counts more than a false alarm).
      rule "RECALL_ENS": average of DistilBERT and TF-IDF >= a per-category threshold (micro-F2).
      rule "BALANCED"  : both models above their own per-category thresholds (micro-F1).
    Version 1 (fallback; the first report's models, fixed 0.5 threshold):
      rule "AND"  : min(TF-IDF, DistilBERT) >= threshold;   rule "TRANS": DistilBERT alone.

    Returns (present, scores):
        present  list of dicts, one per present category, sorted High -> Low then by score:
                 {category, score, threshold, transformer_score, tfidf_score, chance, possible,
                  span_text, highlight, start, risk, reason}
                 possible: the calibrated chance is under POSSIBLE_BELOW (shown as "Possible — check")
        scores   {category: {"transformer": p, "tfidf": p or None, "score": p}}
    """
    rule = rule or default_rule()
    if rule in V2_RULES and not v2_available():
        rule = "TRANS"
    version = "v2" if rule in V2_RULES else "v1"
    trans, best_window = predict_presence(text, progress=progress, version=version)
    if version == "v2":
        tfidf = tfidf_scores(text, "v2")
    else:
        tfidf = tfidf_scores(text) if baseline_available() else {}
        if rule == "AND" and not tfidf:
            rule = "TRANS"

    scores, present = {}, []
    for cat, pt in trans.items():
        pf = tfidf.get(cat)
        if version == "v2":
            hit, sc, thr = _decide_v2(rule, cat, pt, pf)
        else:
            sc = min(pt, pf) if rule == "AND" else pt
            hit, thr = sc >= threshold, threshold
        scores[cat] = {"transformer": pt, "tfidf": pf, "score": sc}
        if hit:
            para, span = locate_clause(cat, best_window[cat], version)
            para, start = _full_paragraph(text, para)
            level, reason = risk_of(cat)
            chance = calibrated(pt) if version == "v2" else None
            present.append({
                "category": cat, "score": sc, "threshold": thr, "transformer_score": pt, "tfidf_score": pf,
                "chance": chance, "possible": chance is not None and chance < POSSIBLE_BELOW,
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


@functools.lru_cache(maxsize=2)
def _load_baseline(version="v1"):
    import pickle
    with open(BASELINE_V2_PATH if version == "v2" else BASELINE_PATH, "rb") as f:
        return pickle.load(f)


def tfidf_scores(text, version="v1"):
    """{category: probability the category is present} from the full-document TF-IDF model."""
    bl = _load_baseline(version)
    X = bl["vec"].transform([text])
    out = {}
    for cat in CATEGORIES:
        kind, obj = bl["classifiers"][cat]
        out[cat] = float(obj) if kind == "const" else float(obj.predict_proba(X)[0, 1])
    return out


# --------------------------------------------------------------------------- #
# Summarizer — fine-tuned FLAN-T5-small. The report (Section 5.1.3) shows it maps
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
_CAL_PATH = os.path.join(BASE, "clause_calibration.json")             # first-setup model
_CAL_V2_PATH = os.path.join(BASE, "clause_calibration_v2.json")       # re-run model (44.9% vs 42.3%)
MIN_CLAUSE_CHARS = 40          # shorter fragments (titles, signature lines) are skipped
AUTO_MAX_CLAUSES = 25          # auto mode: at most this many paragraphs ...
AUTO_MAX_CHARS = 12000         # ... and this much text to be treated as a clause list


# Clause mode, current version: a classifier trained for "which of the 41 types is this clause?", with
# a 42nd class, "none", for paragraphs that are no annotated clause (final project p3/retrain/clause_v2).
# clause_v2.json names the model chosen on the validation contracts and holds its test accuracy; the
# weights live in MODELS_DIR/clause_v2/ (final/ for DistilBERT, tfidf.pkl for the word model). Without
# them, clause mode falls back to the presence model's standardized scores below.
_CLAUSE_V2_CFG = os.path.join(BASE, "clause_v2.json")
CLAUSE_V2 = json.load(open(_CLAUSE_V2_CFG)) if os.path.exists(_CLAUSE_V2_CFG) else None
CLAUSE_V2_DIR = os.path.join(MODELS_DIR, "clause_v2")


def clause_v2_available():
    if not CLAUSE_V2:
        return False
    kind = CLAUSE_V2["model"]
    need = ([os.path.join(CLAUSE_V2_DIR, "final", "config.json")] if kind in ("bert", "average") else []) + \
           ([os.path.join(CLAUSE_V2_DIR, "tfidf.pkl")] if kind in ("tfidf", "average") else [])
    return all(os.path.exists(p) for p in need)


@functools.lru_cache(maxsize=1)
def _load_clause_v2():
    import pickle
    kind, m = CLAUSE_V2["model"], {}
    if kind in ("bert", "average"):
        d = os.path.join(CLAUSE_V2_DIR, "final")
        m["tok"] = AutoTokenizer.from_pretrained(d)
        m["bert"] = AutoModelForSequenceClassification.from_pretrained(d).to(DEVICE).eval()
    if kind in ("tfidf", "average"):
        with open(os.path.join(CLAUSE_V2_DIR, "tfidf.pkl"), "rb") as f:
            m["tfidf"] = pickle.load(f)
    return m


def clause_probs(texts):
    """[len(texts), 42] probabilities over CLAUSE_V2["labels"] (the 41 categories, then "none")."""
    m, kind, parts = _load_clause_v2(), CLAUSE_V2["model"], []
    if "bert" in m:
        enc = m["tok"](texts, truncation=True, max_length=256, padding=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            parts.append(F.softmax(m["bert"](**enc).logits.float(), dim=-1).cpu().numpy())
    if "tfidf" in m:
        parts.append(m["tfidf"]["clf"].predict_proba(m["tfidf"]["vec"].transform(texts)))
    return sum(parts) / len(parts)


def _clause_version():
    """Clause mode uses the re-run model when its files and calibration are present."""
    return "v2" if v2_available() and os.path.exists(_CAL_V2_PATH) else "v1"


def _clause_cal_path():
    return _CAL_V2_PATH if _clause_version() == "v2" else _CAL_PATH


@functools.lru_cache(maxsize=2)
def _calibration(path=None):
    cal = json.load(open(path or _clause_cal_path()))
    assert cal["categories"] == CATEGORIES, "calibration was built for a different category order"
    return np.array(cal["neg_mean"]), np.array(cal["neg_std"]), float(cal["unrecognized_below_z"])


def clause_eval():
    """Held-out accuracy of the model-only clause classifier: {n_test_clauses, acc, top3, risk_level_acc}."""
    if clause_v2_available():
        return CLAUSE_V2["eval"]
    try:
        e = json.load(open(_clause_cal_path()))["eval"]
        return {"n_test_clauses": e["n_test_clauses"], "acc": e["calibrated_acc"], "top3": e["calibrated_top3"],
                "risk_level_acc": e["risk_level_acc"]}
    except (OSError, KeyError, ValueError):
        return None


def clause_mode_available():
    return clause_v2_available() or os.path.exists(_clause_cal_path())


_NUMBERED = re.compile(
    r"^\s*(?:"
    r"(?:section|article|clause)\s+(?P<a>\d+(?:\.\d+)*|[ivxlc]+)[\.\):]?"   # Section 5 / Article IV
    r"|(?P<b>\d+(?:\.\d+)+)[\.\)]?"                                   # 4.2  4.2.  1.3.1
    r"|(?P<e>\d+)[\.\)]"                                                # 1.  3)
    r"|\((?P<c>[a-z0-9]{1,4})\)"                                         # (a)  (iv)  (12)
    r"|(?P<d>[ivxlc]{1,6})[\.\)]"                                         # IV.  ii)
    r")\s", re.IGNORECASE)


def clause_number(paragraph):
    """The paragraph's own clause number ("1", "4.2", "a", "IV"), or None if it is not numbered."""
    m = _NUMBERED.match(paragraph)
    return next((g for g in m.groups() if g), None) if m else None


def split_clauses_detailed(text):
    """
    Split pasted text into clauses: blank-line separated paragraphs, or, if the text has
    no blank lines, lines that start a numbered item (1. / 2) / (a) ...).

    When most paragraphs are numbered clauses, the un-numbered ones (title, preamble,
    recitals, signature block) are not clauses: they are returned separately as skipped.
    Returns (clauses, skipped).
    """
    text = text.replace("\r\n", "\n").strip()
    parts = [p.strip() for p in re.split(r"\n\s*\n", text)]
    if len(parts) == 1:
        parts = [p.strip() for p in
                 re.split(r"\n(?=\s*(?:\d+[\.\)]|\([a-z0-9]+\)|[a-z][\.\)])\s)", text)]
    parts = [p for p in parts if p]
    numbered = [p for p in parts if clause_number(p)]
    if len(numbered) >= 2 and len(numbered) >= 0.5 * len(parts):
        clauses = [p for p in numbered if len(p) >= MIN_CLAUSE_CHARS]
        skipped = [p for p in parts if not clause_number(p)]
    else:
        clauses = [p for p in parts if len(p) >= MIN_CLAUSE_CHARS]
        skipped = []
    return clauses, skipped


def split_clauses(text):
    return split_clauses_detailed(text)[0]


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
        {index, clause, category, risk, reason, source, model_guess, z, prob, presence_score, best_guess, runner_up}
    source is "heading" when the clause heading names the category, else "model".
    category is None (risk "Unrecognized") when no category stands out: with the clause classifier, when
    its most likely class is "none"; with the fallback, when the standardized score z is below the cut-off.
    prob (clause classifier) is the probability of the best of the 41 categories; z is None then.
    """
    if clause_v2_available():
        labels = CLAUSE_V2["labels"]
        idx = [labels.index(c) for c in CATEGORIES]
        out = []
        for i in range(0, len(clauses), 16):
            probs = clause_probs(clauses[i:i + 16])
            for k, cl in enumerate(clauses[i:i + 16]):
                p41 = probs[k][idx]
                order = np.argsort(-p41)
                best, second = int(order[0]), int(order[1])
                is_none = labels[int(probs[k].argmax())] == "none"
                by_heading = heading_category(cl)
                if by_heading:
                    cat, source = by_heading, "heading"
                else:
                    cat, source = (None if is_none else CATEGORIES[best]), "model"
                level, reason = risk_of(cat) if cat else ("Unrecognized", "No clause category stands out clearly — review manually.")
                out.append({
                    "index": i + k + 1, "number": clause_number(cl) or str(i + k + 1),
                    "clause": cl, "category": cat, "risk": level, "reason": reason,
                    "source": source, "model_guess": CATEGORIES[best], "z": None,
                    "prob": float(p41[best]), "none_prob": float(probs[k][labels.index("none")]),
                    "presence_score": float(p41[best]), "best_guess": CATEGORIES[best],
                    "runner_up": CATEGORIES[second], "runner_up_prob": float(p41[second]),
                })
                if progress is not None:
                    progress(len(out) / len(clauses))
        return out
    version = _clause_version()
    tok, model = _load_presence(version)
    mean, std, z_cut = _calibration(_clause_cal_path())
    qs = [CAT_QUESTIONS[c] for c in CATEGORIES]
    out = []
    for i, cl in enumerate(clauses):
        enc = tok(qs, [cl] * len(qs), truncation=True, max_length=_presence_maxlen(version),
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
            "index": i + 1, "number": clause_number(cl) or str(i + 1),
            "clause": cl, "category": cat, "risk": level, "reason": reason,
            "source": source, "model_guess": CATEGORIES[best],
            "z": float(z[best]), "presence_score": float(prob[best]),
            "best_guess": CATEGORIES[best], "runner_up": CATEGORIES[second],
        })
        if progress is not None:
            progress((i + 1) / len(clauses))
    return out
