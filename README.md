# Clausify — Contract Analyzer App

A Streamlit web app that runs the project's trained models live on any contract you give it and
presents the clauses **grouped by risk (High / Medium / Low)**, each with a reason, the quoted
clause and a plain-English line.

> Full write-up: the project report (Section 4.4.9 for the app, Section 5.3.3 for the models it uses).

| Step | Model | Held-out test result (102 CUAD contracts) |
|---|---|---|
| Which of the 41 clause types are present? | DistilBERT reading whole 2,000-character windows (512 tokens), thresholds chosen on a validation split | **Recall-first (default):** finds **90.3%** of High-risk clauses in whole contracts and **88.6%** as the app reads them (below), micro-F1 0.757 · **Balanced ensemble:** micro-F1 **0.809** |
| Where is the clause? | DistilBERT-QA span model, retrained on the 1,200-character chunks it reads | token-F1 **0.779** on a window that holds the clause; **59.6%** of real clauses get a good quote end to end (first span model: 22.1%) |
| Plain-English line | FLAN-T5-small | picks one of 41 category sentences (the report explains why it is not a real summary) |

---

## What you can do

1. **Add a contract**: paste the text, or upload a `.pdf`, `.docx` or `.txt` file.
2. Choose **Whole contract**, **Separate clauses** or **Auto**.
3. Click **Analyze contract** and read the cards, highest risk first. Each card shows the clause type,
   its risk level and reason, the quoted clause with the span model's answer highlighted, and the
   calibrated chance with the model score and the threshold for that clause type. Cards under a 50%
   chance are marked **Possible — check** and listed after the others in their risk group. Below the cards: a highlighted copy of the contract, a missing-protection checklist,
   and PDF/CSV downloads.

**Model settings** (whole-contract mode):

| Setting | Rule | On the test set |
|---|---|---|
| Recall-first (default) | DistilBERT score ≥ its per-type threshold | 90.3% of High-risk clauses found on whole contracts (17 of 176 missed), micro-F1 0.757; 88.6% (20 missed) as the app reads them |
| Recall-first ensemble | average of DistilBERT and TF-IDF ≥ per-type threshold | 86.9% found, micro-F1 0.783; TF-IDF scores short contracts too low |
| Balanced | both models above their own thresholds | micro-F1 0.809, but only 54.0% of High-risk clauses found |

Each card's bar shows a calibrated chance that the clause type is present (isotonic map fitted on the
81 validation contracts; test-set calibration error 0.016), shown between 1% and 99%. The thresholds are
set per clause type and the map is shared, so 37% of the default setting's cards on the test contracts
are under 50%; those are the **Possible — check** cards (30% of them are real clauses, including 46 of
the 159 High-risk clauses found). Clause mode uses a classifier trained for the job (TF-IDF
with logistic regression over 41 clause types plus "none"; `clause_v2.json`): it picks the right one of
41 categories for 72.1% of 721 held-out CUAD clauses, against 44.9% for the earlier presence-model method.

**Long contracts.** Every clause type is checked in the first 30 windows (about 45,500 characters). In a
longer contract the TF-IDF model, applied to each later window, picks the 5 windows most likely to hold
each clause type, and those are checked too. Reading only the first 30 windows found 77.3% of High-risk
clauses on the test contracts; with the 5 extra windows the app finds 88.6%, against 90.3% for reading
every window (`final project p3/retrain/v2_fullwindow/window_cap.py`, measured through this app's code
by `app_presence_test.py`). Parts of a very long contract are still never read.

The thresholds live in `decision_v2.json`, written by the training code
(`final project p3/retrain/v2_fullwindow/tune_and_test.py`), so the app runs exactly the tested setup.
The span model's settings live in `span_v2.json` (chunking, scoring rule, and how the quoted paragraph is
chosen), chosen on the validation contracts by `final project p3/retrain/span_v2/evaluate_span.py`.

---

## How to run

```bash
cd "final project app p3"
pip install -r requirements.txt        # first time only
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. Paste `test_contract.txt` or upload any contract.
A typical contract takes about a minute on a laptop CPU; see the report's speed table.

Tests (a few minutes): `python3 tests/test_app.py`. The output of the last run (46/46 checks passed) is
saved in `tests/test_output.log`, with the date, the code version and the models it used.

---

## Where the models come from

In order: the `CUAD_MODELS_DIR` folder, a `models/` folder here, the sibling
`../final project p3/notebooks/outputs/` folder, or the Hugging Face repo in `CLAUSIFY_HF_REPO`
(default `af123Af/clausify-models`). The folder needs:

```
presence_v2/final/      re-run presence model (512 tokens)         ← detection and clause mode
presence_mil/final/     first-setup presence model (256 tokens)    ← fallback only
span_v2/final/          retrained span model                       ← finds and quotes the clause
span/final/             first span model                           ← fallback only
clause_v2/tfidf.pkl     clause classifier                          ← clause mode
summarizer/final/       FLAN-T5 summarizer
```

plus the TF-IDF models (`baseline_v2/baseline.pkl` and `baseline/baseline.pkl`, or `baseline_v2.pkl`
and `baseline.pkl` in `../artifacts/` locally). If `presence_v2` or `baseline_v2` is missing, the app
falls back to the first setup's models and says so in the settings panel. If `span_v2` is missing, the
app quotes the paragraph the presence model scores highest and highlights the first span model's answer.

---

*Research prototype, not legal advice. Risk levels are per clause category, judged for the party with
less bargaining power.*
