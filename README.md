# Clausify — Contract Analyzer App

A Streamlit web app that runs the project's trained models live on any contract you give it and
presents the clauses **grouped by risk (High / Medium / Low)**, each with a reason, the quoted
clause and a plain-English line.

> Full write-up: the project report (Section 4.6 for the app, Section 5.3.3 for the models it uses).

| Step | Model | Held-out test result (102 CUAD contracts) |
|---|---|---|
| Which of the 41 clause types are present? | DistilBERT reading whole 2,000-character windows (512 tokens), thresholds chosen on a validation split | **Recall-first (default):** finds **90.3%** of High-risk clauses, micro-F1 0.757 · **Balanced ensemble:** micro-F1 **0.809** |
| Where is the clause? | DistilBERT-QA span model | token-F1 **0.764** when given the right window |
| Plain-English line | FLAN-T5-small | picks one of 41 category sentences (the report explains why it is not a real summary) |

---

## What you can do

1. **Add a contract**: paste the text, or upload a `.pdf`, `.docx` or `.txt` file.
2. Choose **Whole contract**, **Separate clauses** or **Auto**.
3. Click **Analyze contract** and read the cards, highest risk first. Each card shows the clause type,
   its risk level and reason, the quoted clause, and the model score next to the threshold for that
   clause type. Below the cards: a highlighted copy of the contract, a missing-protection checklist,
   and PDF/CSV downloads.

**Model settings** (whole-contract mode):

| Setting | Rule | On the test set |
|---|---|---|
| Recall-first (default) | DistilBERT score ≥ its per-type threshold | 90.3% of High-risk clauses found (17 of 176 missed), micro-F1 0.757 |

Each card's bar shows a calibrated chance that the clause type is present (isotonic map fitted on the
81 validation contracts; test-set calibration error 0.016), shown between 1% and 99%. Clause mode picks
the right one of 41 categories for 44.9% of held-out CUAD clauses (`clause_calibration_v2.json`).
| Recall-first ensemble | average of DistilBERT and TF-IDF ≥ per-type threshold | 86.9% found, micro-F1 0.783; TF-IDF scores short contracts too low |
| Balanced | both models above their own thresholds | micro-F1 0.809, but only 54.0% of High-risk clauses found |

The thresholds live in `decision_v2.json`, written by the training code
(`final project p3/retrain/v2_fullwindow/tune_and_test.py`), so the app runs exactly the tested setup.

---

## How to run

```bash
cd "final project app p3"
pip install -r requirements.txt        # first time only
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. Paste `test_contract.txt` or upload any contract.
A typical contract takes about a minute on a laptop CPU; see the report's speed table.

Tests (a few minutes): `python3 tests/test_app.py`

---

## Where the models come from

In order: the `CUAD_MODELS_DIR` folder, a `models/` folder here, the sibling
`../final project p3/notebooks/outputs/` folder, or the Hugging Face repo in `CLAUSIFY_HF_REPO`
(default `af123Af/clausify-models`). The folder needs:

```
presence_v2/final/      re-run presence model (512 tokens)         ← detection and clause mode
presence_mil/final/     first-setup presence model (256 tokens)    ← fallback only
span/final/             span model
summarizer/final/       FLAN-T5 summarizer
```

plus the TF-IDF models (`baseline_v2/baseline.pkl` and `baseline/baseline.pkl`, or `baseline_v2.pkl`
and `baseline.pkl` in `../artifacts/` locally). If `presence_v2` or `baseline_v2` is missing, the app
falls back to the first setup's models and says so in the settings panel.

---

*Research prototype, not legal advice. Risk levels are per clause category, judged for the party with
less bargaining power.*
