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
# Paths — models live in the sibling "final project p3" deliverable folder.
# Override with the CUAD_MODELS_DIR environment variable if you move them.
# --------------------------------------------------------------------------- #
BASE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODELS = os.path.join(
    os.path.dirname(BASE), "final project p3", "notebooks", "outputs"
)
MODELS_DIR = os.environ.get("CUAD_MODELS_DIR", _DEFAULT_MODELS)
PRESENCE_DIR = os.path.join(MODELS_DIR, "presence_mil", "final")
SPAN_DIR = os.path.join(MODELS_DIR, "span", "final")

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


def analyze(text, threshold=0.5, progress=None):
    """
    Full pipeline for one contract.

    Returns a list of dicts (one per PRESENT category), sorted by confidence:
        {category, presence_score, span_text, span_conf}
    plus the raw presence scores for every category.
    """
    scores, best_window = predict_presence(text, progress=progress)

    present = []
    for cat, sc in scores.items():
        if sc >= threshold:
            span, span_conf = extract_span(CAT_QUESTIONS[cat], best_window[cat])
            level, reason = risk_of(cat)
            present.append({
                "category": cat,
                "presence_score": sc,
                "span_text": span,
                "span_conf": span_conf,
                "risk": level,
                "reason": reason,
            })

    # sort by risk (High → Low), then by confidence within a risk band
    present.sort(key=lambda d: (RISK_ORDER[d["risk"]], -d["presence_score"]))
    return present, scores
