# Clausify — Contract Analyzer App

A Streamlit web app that **loads the two trained models** and runs them live on any
contract you give it, then presents the clauses **grouped by risk (High / Medium / Low)**.
Built to show that both models perform well on unseen contracts.

> Full write-up of how the app works: **`APP_REPORT.md` / `APP_REPORT.pdf`**.

| Model | What it does | Held-out test score (80/20) |
|---|---|---|
| **Model 1 — Presence Classifier** (DistilBERT) | For each of 41 clause categories, is it in this contract? | Accuracy **85.49%**, F1 **0.776** |
| **Model 2 — Span Extractor** (DistilBERT-QA) | Pull out the exact clause text | token-F1 **0.763**, 82.7% overlap |

---

## What you can do

1. **Add a contract** two ways:
   - **① Upload** a `.txt` contract file, **or**
   - **② Paste** the contract text into the box.
2. Click **Analyze contract**.
3. See **both models' results side by side**:
   - **Model 1** lists every clause category it detects, with a confidence bar.
   - **Model 2** shows the exact clause text it extracted for each detected category.
   - An expandable table shows all 41 categories with their scores.

A sample contract, `sample_contract.txt`, is included so you can test immediately.

---

## How to run

```bash
cd "final project app p3"
pip install -r requirements.txt        # first time only
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. Upload `sample_contract.txt`
(or paste any contract) and click **Analyze contract**.

> First analysis takes a few seconds while the models load; after that each
> contract takes roughly 5–15 seconds on a laptop CPU.

---

## Files

```
final project app p3/
├── app.py                 — the Streamlit UI (two input bars + two result panels)
├── model_utils.py         — loads both models, windows the contract, runs inference
├── cat_questions.json     — the 41 category → CUAD-question strings the models expect
├── sample_contract.txt    — a real CUAD contract to test with
├── requirements.txt
└── README.md              — this file
```

## Where the models come from

The app loads the trained weights from the sibling deliverable folder:

```
../final project p3/notebooks/outputs/presence_mil/final/   ← Model 1
../final project p3/notebooks/outputs/span/final/           ← Model 2
```

If you move the models, point the app at them:

```bash
CUAD_MODELS_DIR="/path/to/outputs" streamlit run app.py
```

(That folder must contain `presence_mil/final/` and `span/final/`.)

---

## How it works (matches the training notebook)

- The contract is sliced into **2000-character windows** (1500 stride), exactly as in training.
- **Model 1** scores every `(category-question, window)` pair and **max-pools** over
  windows: a category is *present* if any window fires above the threshold.
- **Model 2** then runs question-answering on the best window for each detected
  category to extract the clause text.
- Everything runs on **CPU** for portability (no GPU required).

The models were downsized for laptop training (DistilBERT rather than DeBERTa-v3),
so span boundaries are approximate — the point of the demo is that both models
generalize to contracts they were never trained on.
