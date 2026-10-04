"""
Calibrate and evaluate the app's clause-by-clause mode.

The presence model scores (category question, text) pairs. Applied to a single
clause, some categories score high on almost anything (e.g. "Parties" fires on
any text that names the two sides), so the raw arg-max picks the wrong category.
We therefore standardize each category's score against how that category scores
on genuine CUAD clauses of OTHER categories:

    z(c | clause) = (logit p(c | clause) - mean_neg[c]) / std_neg[c]

and classify the clause as arg-max_c z.  The per-category statistics come from
TRAINING-split clauses only; accuracy is then measured on TEST-split clauses.

Writes clause_calibration.json next to this file.
    python calibrate_clause_mode.py            (~10-15 min on CPU)
"""
import json, os, random, time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

import model_utils as mu

SEED, PER_CAT_TRAIN, PER_CAT_TEST, MAXLEN = 42, 40, 20, 256
CUAD = os.path.expanduser("~/p3_notebook/CUADv1.json")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
CLAUSES_PER_BATCH = 2


def clauses_by_split():
    js = json.load(open(CUAD))["data"]
    titles = [c["title"] for c in js]
    perm = np.random.default_rng(SEED).permutation(len(titles))
    test = {titles[i] for i in perm[:int(0.2 * len(titles))]}       # same split as the project
    out = {"train": defaultdict(list), "test": defaultdict(list)}
    for c in js:
        split = "test" if c["title"] in test else "train"
        for qa in c["paragraphs"][0]["qas"]:
            cat = qa["id"].rsplit("__", 1)[-1]
            for a in qa["answers"]:
                t = a["text"].strip()
                if len(t) >= 20:                       # skip bare names/dates fragments
                    out[split][cat].append(t)
    return out


def sample(d, k, rng):
    return [(cat, t) for cat in mu.CATEGORIES for t in rng.sample(d[cat], min(k, len(d[cat])))]


def logits_all_categories(texts):
    """Return an array [len(texts), 41] of log-odds that each category is present."""
    tok, model = mu._load_presence()
    model = model.to(DEV)
    qs = [mu.CAT_QUESTIONS[c] for c in mu.CATEGORIES]
    out = np.zeros((len(texts), len(qs)))
    for i in range(0, len(texts), CLAUSES_PER_BATCH):
        chunk = texts[i:i + CLAUSES_PER_BATCH]
        enc = tok(qs * len(chunk), [t for t in chunk for _ in qs], truncation=True,
                  max_length=MAXLEN, padding=True, return_tensors="pt").to(DEV)
        with torch.no_grad():
            lg = model(**enc).logits.float().cpu()
        out[i:i + len(chunk)] = (lg[:, 1] - lg[:, 0]).numpy().reshape(len(chunk), len(qs))
        if i % 300 == 0:
            print(f"  {i}/{len(texts)}", flush=True)
    model.to(mu.DEVICE)
    return out


def main():
    rng = random.Random(SEED)
    data = clauses_by_split()
    tr, te = sample(data["train"], PER_CAT_TRAIN, rng), sample(data["test"], PER_CAT_TEST, rng)
    print(f"train clauses {len(tr)}, test clauses {len(te)}")
    cats = mu.CATEGORIES
    t0 = time.time()
    Ltr = logits_all_categories([t for _, t in tr])
    ytr = np.array([cats.index(c) for c, _ in tr])
    mean, std = np.zeros(len(cats)), np.ones(len(cats))
    for j in range(len(cats)):
        neg = Ltr[ytr != j, j]
        mean[j], std[j] = neg.mean(), max(neg.std(), 1e-3)

    Lte = logits_all_categories([t for _, t in te])
    yte = np.array([cats.index(c) for c, _ in te])
    raw = (Lte.argmax(1) == yte).mean()
    Z = (Lte - mean) / std
    zacc = (Z.argmax(1) == yte).mean()
    top3 = np.mean([yte[i] in np.argsort(-Z[i])[:3] for i in range(len(yte))])
    risk = lambda j: mu.risk_of(cats[j])[0]
    risk_acc = np.mean([risk(Z[i].argmax()) == risk(yte[i]) for i in range(len(yte))])
    print(f"done in {(time.time()-t0)/60:.1f} min")
    print(f"TEST clause accuracy  raw arg-max {raw:.3f} | calibrated {zacc:.3f} | "
          f"calibrated top-3 {top3:.3f} | risk level correct {risk_acc:.3f}")

    # z-score of the TRUE category on test clauses -> sets the "unrecognized" cut-off
    ztrue = Z[np.arange(len(yte)), yte]
    zmax = Z.max(1)
    json.dump({
        "categories": cats,
        "neg_mean": mean.round(4).tolist(), "neg_std": std.round(4).tolist(),
        "unrecognized_below_z": 2.0,
        "eval": {"n_test_clauses": int(len(yte)), "raw_argmax_acc": round(float(raw), 4),
                 "calibrated_acc": round(float(zacc), 4), "calibrated_top3": round(float(top3), 4),
                 "risk_level_acc": round(float(risk_acc), 4),
                 "share_test_max_z_below_2": round(float((zmax < 2.0).mean()), 4),
                 "median_true_z": round(float(np.median(ztrue)), 2)},
        "notes": "Stats from TRAIN-split CUAD clauses (40 per category); accuracy on TEST-split "
                 "clauses (20 per category). Split = project seed-42 contract-level 80/20.",
    }, open(os.path.join(mu.BASE, "clause_calibration.json"), "w"), indent=1)
    print("wrote clause_calibration.json")


if __name__ == "__main__":
    main()
