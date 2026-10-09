"""
Clausify — premium Streamlit demo (CUAD contract analysis).

Add a contract (upload a .txt file OR paste the text) and see BOTH trained
models work on it, live:

  Model 1 — Presence Classifier : which of the 41 clause categories are present
  Model 2 — Span Extractor      : the exact clause text pulled from the contract

Run:  streamlit run app.py
"""

from features import escape  # HTML-escape + '$' and newline safe for st.markdown

import streamlit as st

import model_utils as mu
import features as fx

st.set_page_config(page_title="Clausify", page_icon="⚖️", layout="wide")

# --------------------------------------------------------------------------- #
# Premium styling — animated gradient background, glass panels, custom result
# cards. All injected as CSS/HTML.
# --------------------------------------------------------------------------- #
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');

:root{
  --bg0:#070a18; --bg1:#0d1230; --ink:#e8ecff; --muted:#9aa3c7;
  --accent1:#7c5cff; --accent2:#22d3ee; --accent3:#f472b6;
  --glass:rgba(255,255,255,.06); --glass-brd:rgba(255,255,255,.12);
}

/* ---- animated background ---- */
.stApp{
  background:
    radial-gradient(1200px 700px at 12% -10%, rgba(124,92,255,.35), transparent 60%),
    radial-gradient(1000px 700px at 110% 10%, rgba(34,211,238,.22), transparent 55%),
    radial-gradient(900px 800px at 50% 120%, rgba(244,114,182,.20), transparent 55%),
    linear-gradient(160deg, var(--bg0), var(--bg1));
  color:var(--ink);
  font-family:'Inter',system-ui,sans-serif;
}
.stApp::before{
  content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background:
    radial-gradient(circle at 20% 30%, rgba(124,92,255,.20), transparent 12%),
    radial-gradient(circle at 80% 60%, rgba(34,211,238,.16), transparent 12%),
    radial-gradient(circle at 60% 20%, rgba(244,114,182,.14), transparent 10%);
  filter:blur(30px); animation:float 18s ease-in-out infinite alternate;
}
@keyframes float{
  0%{transform:translate3d(0,0,0) scale(1)}
  100%{transform:translate3d(0,-20px,0) scale(1.06)}
}
[data-testid="stHeader"]{background:transparent;}
.block-container{position:relative; z-index:1; padding-top:2.2rem; max-width:1180px;}

/* ---- hero ---- */
.hero{ text-align:center; margin:.4rem 0 1.6rem; }
.hero .badge{
  display:inline-flex; align-items:center; gap:.5rem;
  padding:.35rem .9rem; border-radius:999px; font-size:.78rem; font-weight:600;
  color:#cdd4ff; background:var(--glass); border:1px solid var(--glass-brd);
  backdrop-filter:blur(10px); letter-spacing:.04em; text-transform:uppercase;
}
.hero h1{
  font-family:'Sora',sans-serif; font-weight:800; font-size:3.1rem;
  margin:.7rem 0 .3rem; line-height:1.05;
  background:linear-gradient(100deg,#fff 10%,#b9a5ff 40%,#7cf3ff 70%,#f8a8d8 95%);
  background-size:200% auto; -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent; animation:shine 6s linear infinite;
}
@keyframes shine{to{background-position:200% center}}
.hero p{ color:var(--muted); font-size:1.02rem; margin:0; }

/* ---- section titles ---- */
.sec{ font-family:'Sora',sans-serif; font-weight:700; font-size:1.15rem;
  margin:.2rem 0 .8rem; display:flex; align-items:center; gap:.55rem; }
.sec .dot{ width:10px; height:10px; border-radius:3px; }
.dot.blue{ background:linear-gradient(135deg,#7c5cff,#22d3ee); box-shadow:0 0 14px #7c5cff; }
.dot.green{ background:linear-gradient(135deg,#34d399,#22d3ee); box-shadow:0 0 14px #34d399; }

/* ---- glass panels ---- */
.glass{
  background:var(--glass); border:1px solid var(--glass-brd); border-radius:20px;
  padding:1.1rem 1.2rem; backdrop-filter:blur(14px);
  box-shadow:0 20px 50px rgba(0,0,0,.35);
}

/* ---- inputs ---- */
.stTextArea textarea, [data-testid="stFileUploaderDropzone"]{
  background:rgba(255,255,255,.05) !important; color:var(--ink) !important;
  border:1px solid var(--glass-brd) !important; border-radius:16px !important;
  backdrop-filter:blur(10px);
}
.stTextArea textarea::placeholder{ color:#7681ad; }
[data-testid="stFileUploaderDropzone"]{ padding:1rem !important; }
.stSlider label, .stTextArea label, .stFileUploader label{ color:var(--muted) !important; }

/* ---- primary button ---- */
.stButton>button{
  border:none; border-radius:14px; padding:.7rem 1.4rem; font-weight:700;
  color:#0b0f24; background:linear-gradient(100deg,#8b7bff,#37e0f0);
  box-shadow:0 10px 30px rgba(124,92,255,.45); transition:.18s ease;
}
.stButton>button:hover{ transform:translateY(-2px); box-shadow:0 16px 40px rgba(55,224,240,.5); }
.stButton>button:disabled{ opacity:.45; box-shadow:none; }

/* ---- Model 1 confidence bars ---- */
.bar-row{ margin:.55rem 0; }
.bar-label{ display:flex; justify-content:space-between; font-size:.92rem;
  font-weight:600; margin-bottom:.28rem; }
.bar-label .pct{ color:#8ff0ff; font-variant-numeric:tabular-nums; }
.bar-track{ height:9px; border-radius:99px; background:rgba(255,255,255,.08); overflow:hidden; }
.bar-fill{ height:100%; border-radius:99px;
  background:linear-gradient(90deg,#7c5cff,#22d3ee,#a78bfa);
  background-size:200% 100%; animation:slide 3s linear infinite;
  box-shadow:0 0 12px rgba(34,211,238,.55); }
@keyframes slide{to{background-position:200% 0}}

/* ---- Model 2 span cards ---- */
.span-card{
  background:rgba(255,255,255,.045); border:1px solid var(--glass-brd);
  border-left:3px solid #34d399; border-radius:14px; padding:.75rem .9rem; margin:.55rem 0;
}
.span-cat{ font-weight:700; font-size:.92rem; color:#c9f7e6; margin-bottom:.3rem; }
.span-text{ color:#e8ecff; font-size:.92rem; line-height:1.45; }
.span-conf{ color:var(--muted); font-size:.76rem; margin-top:.4rem; }
.empty{ color:var(--muted); font-style:italic; }

/* ---- summary chip ---- */
.chip{ display:inline-block; padding:.4rem .9rem; border-radius:999px; font-weight:700;
  background:var(--glass); border:1px solid var(--glass-brd); color:#d7ddff; }

/* streamlit expander */
[data-testid="stExpander"]{ border:1px solid var(--glass-brd) !important;
  border-radius:14px !important; background:rgba(255,255,255,.03) !important; }

/* ---- risk groups & clause cards ---- */
.risk-summary{ display:flex; gap:.7rem; flex-wrap:wrap; margin:.2rem 0 1.3rem; }
.risk-pill{ display:inline-flex; align-items:center; gap:.5rem; padding:.5rem 1rem;
  border-radius:999px; font-weight:700; font-size:.92rem; border:1px solid var(--glass-brd);
  background:var(--glass); backdrop-filter:blur(10px); }
.risk-pill .n{ font-family:'Sora',sans-serif; font-size:1.05rem; }

.grp-head{ font-family:'Sora',sans-serif; font-weight:700; font-size:1.05rem;
  display:flex; align-items:center; gap:.6rem; margin:1.3rem 0 .7rem; }
.grp-head .tag{ padding:.2rem .7rem; border-radius:8px; font-size:.78rem; letter-spacing:.05em; }
.grp-head .count{ color:var(--muted); font-weight:600; font-size:.9rem; }

.clause{ background:rgba(255,255,255,.045); border:1px solid var(--glass-brd);
  border-left:4px solid; border-radius:14px; padding:.85rem 1rem; margin:.6rem 0; }
.clause .top{ display:flex; justify-content:space-between; align-items:center; gap:.6rem; }
.clause .name{ font-weight:700; font-size:1rem; }
.clause .badge2{ white-space:nowrap; flex-shrink:0; font-size:.72rem; font-weight:800; letter-spacing:.06em;
  padding:.22rem .6rem; border-radius:7px; }
.clause .reason{ color:var(--muted); font-size:.86rem; margin:.35rem 0 .55rem; }
.clause .track{ height:7px; border-radius:99px; background:rgba(255,255,255,.08); overflow:hidden; }
.clause .fill{ height:100%; border-radius:99px; }
.clause .conf{ font-size:.76rem; color:#9fb0e6; margin-top:.3rem; }
.clause .snippet{ margin-top:.6rem; padding:.55rem .7rem; border-radius:10px;
  background:rgba(0,0,0,.22); font-size:.86rem; line-height:1.45; color:#dfe5ff; }
.clause .snippet mark{ background:rgba(124,92,255,.35); color:#fff; border-radius:3px; padding:0 .1rem; }
.clause .snippet .lbl{ display:block; font-size:.68rem; letter-spacing:.08em; text-transform:uppercase;
  color:#7f8bbd; margin-bottom:.25rem; }

.clause.possible{ border-left-style:dashed; background:rgba(255,255,255,.025); }
.clause .maybe{ font-size:.68rem; font-weight:800; letter-spacing:.06em; padding:.18rem .5rem;
  border-radius:7px; background:rgba(138,147,184,.2); color:#d3d8ee; margin-right:.35rem; white-space:nowrap; }
.grp-sub{ color:var(--muted); font-size:.84rem; font-weight:600; margin:.9rem 0 .2rem .2rem; }

/* risk colors */
.high{  --c:#ff5a7a; }  .med{ --c:#f7b955; }  .low{ --c:#34d399; }
.clause.high{ border-left-color:#ff5a7a; }
.clause.med{  border-left-color:#f7b955; }
.clause.low{  border-left-color:#34d399; }
.clause.high .fill{ background:linear-gradient(90deg,#ff5a7a,#ff8a5a); }
.clause.med  .fill{ background:linear-gradient(90deg,#f7b955,#ffd27a); }
.clause.low  .fill{ background:linear-gradient(90deg,#34d399,#22d3ee); }
.badge2.high{ background:rgba(255,90,122,.18); color:#ff98ac; }
.badge2.med{  background:rgba(247,185,85,.18); color:#ffcf8a; }
.badge2.low{  background:rgba(52,211,153,.18); color:#7ff0c8; }
.tag.high{ background:rgba(255,90,122,.18); color:#ff98ac; }
.tag.med{  background:rgba(247,185,85,.18); color:#ffcf8a; }
.tag.low{  background:rgba(52,211,153,.18); color:#7ff0c8; }
.pill-high{ color:#ff98ac; } .pill-med{ color:#ffcf8a; } .pill-low{ color:#7ff0c8; } .pill-unk{ color:#c3c9e0; }
.clause.unk{ border-left-color:#8a93b8; } .clause.unk .fill{ background:#8a93b8; }
.badge2.unk, .tag.unk{ background:rgba(138,147,184,.18); color:#c3c9e0; }

/* input-mode toggle (radio as segmented control) */
[data-testid="stRadio"] > div{ gap:.6rem; }
[data-testid="stRadio"] label{
  background:var(--glass); border:1px solid var(--glass-brd); border-radius:12px;
  padding:.45rem 1rem !important; backdrop-filter:blur(10px); transition:.16s ease;
  color:var(--ink) !important; font-weight:600;
}
[data-testid="stRadio"] label:hover{ border-color:rgba(124,92,255,.6); }
[data-testid="stRadio"] label:has(input:checked){
  background:linear-gradient(100deg,rgba(124,92,255,.35),rgba(34,211,238,.25));
  border-color:rgba(124,92,255,.8); box-shadow:0 8px 24px rgba(124,92,255,.35);
}
[data-testid="stRadio"] label > div:first-child{ display:none; }  /* hide the dot */

/* plain-English line, jump link, contract view, checklist */
.clause .pe{ margin:.45rem 0 .1rem; font-size:.9rem; color:#e9ecff; }
.clause .pe .lbl2{ font-size:.66rem; letter-spacing:.08em; text-transform:uppercase; color:#8f9bd1; margin-right:.35rem; }
.clause .jump{ display:inline-block; margin-top:.45rem; font-size:.78rem; color:#9fb6ff; text-decoration:none; }
.clause .jump:hover{ text-decoration:underline; }
.contract-view{ max-height:520px; overflow:auto; white-space:pre-wrap; line-height:1.55; font-size:.88rem;
  padding:1rem 1.1rem; border-radius:14px; background:rgba(0,0,0,.25); border:1px solid var(--glass-brd); color:#dfe5ff; }
.contract-view mark.hl{ border-radius:3px; padding:0 .1rem; scroll-margin-top:90px; }
.contract-view .hl-tag{ font-size:.62rem; font-weight:700; letter-spacing:.03em; }
.check{ display:flex; gap:.6rem; align-items:flex-start; padding:.55rem .2rem; border-bottom:1px solid rgba(255,255,255,.06); }
.check .ic{ font-size:1rem; } .check .nm{ font-weight:700; } .check .wy{ color:var(--muted); font-size:.86rem; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="hero">
      <h1>Clausify</h1>
      <p>Analyze a whole contract or a list of clauses — trained models find each clause type and rank it by risk.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "⚠️ Research prototype — **not legal advice**. Risk levels are per clause *category*, scored "
    "for the party with less bargaining power. The risk "
    "badge and category are more reliable than the quoted text, which may not be the exact clause "
    "— read each card as \"look here\". Each bar shows the chance that the clause type is present, "
    "calibrated on held-out CUAD contracts; on contracts unlike CUAD's it is only a guide. Cards under "
    "50% are marked **Possible — check** and listed after the others in their risk group. "
    "Only the first 30 windows (about 45,500 characters) of a contract are scanned."
)

# --------------------------------------------------------------------------- #
# Guard
# --------------------------------------------------------------------------- #
if not mu.models_available():
    st.error(
        "Trained models not found.\n\n"
        f"Expected them under: `{mu.MODELS_DIR}`\n\n"
        "Set the `CUAD_MODELS_DIR` environment variable to the folder that "
        "contains `presence_mil/final/` and `span/final/`, or make sure the Hugging Face "
        f"repo `{mu.HF_REPO}` is reachable so they can be downloaded."
    )
    st.stop()

# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
st.markdown('<div class="sec"><span class="dot blue"></span>Add a contract</div>',
            unsafe_allow_html=True)

mode = st.radio("Choose how to add your contract",
                ["📋 Paste text", "📁 Upload a file"],
                horizontal=True, label_visibility="collapsed")

contract_text, source = "", None
if mode == "📋 Paste text":
    pasted = st.text_area("Paste", height=180, label_visibility="collapsed",
                          placeholder="Paste a full contract, or a list of clauses separated by blank lines…")
    if pasted.strip():
        contract_text, source = pasted.strip(), "pasted text"
else:
    uploaded = st.file_uploader("Upload a contract — .pdf, .docx or .txt", type=["pdf", "docx", "txt"])
    if uploaded is not None:
        try:
            contract_text, note = fx.extract_text(uploaded.name, uploaded.getvalue())
            contract_text = contract_text.strip()
            source = f"uploaded file · {uploaded.name}"
            if note:
                st.warning(note)
            if contract_text:
                st.caption(f"Read {len(contract_text):,} characters from {uploaded.name}.")
        except Exception as e:                      # corrupt or password-protected file
            st.error(f"Could not read {uploaded.name}: {e}")

analyze_as = st.radio(
    "Analyze as",
    ["✨ Auto", "📄 Whole contract", "🧩 Separate clauses"],
    horizontal=True, label_visibility="collapsed",
    help="Separate clauses: every paragraph is one clause and gets exactly one category and risk "
         "level. Whole contract: the models scan the document for all 41 clause types. Auto picks "
         "Separate clauses for a short list of paragraphs.",
)
with st.expander("⚙️ Model settings"):
    rule_opts = mu.rule_options()
    if mu.v2_available():
        rule_help = ("On the 102 CUAD test contracts: Recall-first (default) finds 90.3% of High-risk "
                     "clauses (misses 17 of 176), micro-F1 0.757. The recall-first ensemble finds 86.9% "
                     "with fewer false alarms (micro-F1 0.783), but its TF-IDF half learnt from long SEC "
                     "filings and scores short contracts too low. Balanced has the highest micro-F1 (0.809) "
                     "but finds only 54.0% of High-risk clauses. All settings were chosen on a separate "
                     "validation split, with one threshold per clause type.")
    else:
        rule_help = ("Version-2 model files not found, so the original models are used. DistilBERT alone "
                     "misses 15.9% of High-risk clauses on the CUAD test set, the AND-ensemble 36.9%.")
    rule = st.radio("Whole-contract detection", rule_opts, format_func=lambda r: mu.RULES[r],
                    help=rule_help)
    use_summ = st.checkbox("Add a plain-English line to each clause (FLAN-T5 summarizer)",
                           value=mu.summarizer_available(), disabled=not mu.summarizer_available())
threshold = 0.5   # version-1 fallback only; version 2 uses the per-category thresholds in decision_v2.json
go = st.button("🔍  Analyze contract", type="primary", disabled=not contract_text)

# --------------------------------------------------------------------------- #
# Analysis — results are kept in session state so download buttons and
# reruns do not wipe them.
# --------------------------------------------------------------------------- #
_CLS = {"High": "high", "Medium": "med", "Low": "low", "Unrecognized": "unk"}
LEVELS = ("High", "Medium", "Low", "Unrecognized")


def _pct(x):
    return "—" if x is None else f"{x*100:.0f}%"


QUOTE_LABEL = ("Located clause · the span model finds the clause and highlights it; its paragraph is quoted"
               if mu.span_v2_available() and mu.SPAN_V2["quote"] == "span_paragraph"
               else "Located clause · DistilBERT picks the paragraph, the span model highlights the key phrase")


def run_analysis(text, src):
    use_clauses = mu.clause_mode_available() and (
        analyze_as == "🧩 Separate clauses"
        or (analyze_as == "✨ Auto" and mu.looks_like_clause_list(text)))
    clauses, skipped = mu.split_clauses_detailed(text) if use_clauses else ([], [])
    if use_clauses and not clauses:
        st.warning("No clauses of at least 40 characters were found; analyzing as a whole contract.")
        use_clauses = False

    prog = st.progress(0.0, text="Reading the contract…")
    upd = lambda label: (lambda f: prog.progress(min(1.0, f), text=f"{label}… {f*100:.0f}%"))
    with st.spinner("Loading models…"):
        mu.warm_up()

    rows = []
    if use_clauses:
        items = mu.classify_clauses(clauses, progress=upd("Classifying each clause"))
        for it in items:
            z = it["z"]
            if it["category"] and it["source"] == "heading":
                agree = "model agrees" if it["model_guess"] == it["category"] else f"model's own guess: {it['model_guess']}"
                bar, label = 100, f"Category from the clause heading ({agree})"
            elif it["category"]:
                strength = "strong" if z >= 4 else "moderate" if z >= 2.5 else "weak"
                bar = int(max(5, min(100, z / 6 * 100)))
                label = f"Category match (model): {strength} (score {z:.1f}; runner-up: {it['runner_up']})"
            else:
                bar = int(max(5, min(100, z / 6 * 100)))
                label = f"Closest category: {it['best_guess']} — match too weak to assign (score {z:.1f})"
            rows.append({
                "#": it["number"], "category": it["category"] or "Unrecognized", "risk": it["risk"],
                "risk_reason": it["reason"], "decided_by": it["source"], "score": round(z, 2),
                "transformer_score": round(it["presence_score"], 3), "tfidf_score": None,
                "quoted_text": it["clause"], "quote": it["clause"], "start": text.find(it["clause"]),
                "highlight": "", "title": f'Clause {it["number"]} · {it["category"] or "No clear category"}',
                "label": it["category"] or "Unrecognized", "bar": bar, "bar_label": label,
                "snippet_label": "Your clause", "score_text": label,
            })
        mode_name = "separate clauses"
    else:
        present, scores = mu.analyze(text, threshold=threshold, rule=rule,
                                     progress=upd("Scanning for the 41 clause types"))
        decided_by = {"RECALL": "Recall-first DistilBERT", "RECALL_ENS": "Recall-first ensemble",
                      "BALANCED": "Balanced ensemble",
                      "AND": "AND-ensemble (original)", "TRANS": "DistilBERT (original)"}[rule]
        for k, it in enumerate(present, 1):
            pt, pf = it["transformer_score"], it["tfidf_score"]
            maybe = "Possible — check. " if it.get("possible") else ""
            if rule == "RECALL":
                label = (f"About {_pct(it['chance'])} chance this clause type is present "
                         f"(DistilBERT score {_pct(pt)}, above the {_pct(it['threshold'])} set for this type)")
            elif rule == "RECALL_ENS":
                label = (f"Combined score {_pct(it['score'])}, above the {_pct(it['threshold'])} set for this "
                         f"clause type (DistilBERT {_pct(pt)}, TF-IDF {_pct(pf)})")
            elif rule == "BALANCED":
                label = f"Both models above their thresholds for this clause type (DistilBERT {_pct(pt)}, TF-IDF {_pct(pf)})"
            elif rule == "AND":
                label = f"Ensemble score {_pct(it['score'])} (DistilBERT {_pct(pt)}, TF-IDF {_pct(pf)})"
            else:
                label = f"DistilBERT score {_pct(pt)} (TF-IDF {_pct(pf)}, not used for detection)"
            rows.append({
                "#": k, "category": it["category"], "risk": it["risk"], "risk_reason": it["reason"],
                "decided_by": decided_by, "score": round(it["score"], 3),
                "transformer_score": round(pt, 3), "tfidf_score": None if pf is None else round(pf, 3),
                "quoted_text": it["span_text"], "quote": it["span_text"], "start": it["start"],
                "highlight": it["highlight"], "title": it["category"], "label": it["category"],
                "bar": round((it["chance"] if it.get("chance") is not None else it["score"]) * 100),
                "bar_label": maybe + label, "score_text": maybe + label,
                "possible": bool(it.get("possible")), "status": "Possible — check" if it.get("possible") else "Found",
                "snippet_label": QUOTE_LABEL,
            })
        mode_name = "whole contract · " + decided_by

    if use_summ and rows:
        summaries = mu.summarize([r["quote"] for r in rows], progress=upd("Writing plain-English lines"))
        for r, s in zip(rows, summaries):
            r["plain_english"] = s
            sc = mu.summary_category(s)
            if sc and r["risk"] != "Unrecognized" and sc != r["category"]:
                r["summary_mismatch"] = sc      # the summarizer read this text as a different category
    prog.empty()

    found = [r["category"] for r in rows if r["risk"] != "Unrecognized"]
    st.session_state["result"] = {
        "rows": rows, "text": text, "source": src, "mode": mode_name,
        "clause_mode": use_clauses, "missing": mu.missing_protections(found),
        "skipped": skipped if use_clauses else [],
    }


if go and contract_text:
    run_analysis(contract_text, source)

# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def card(r, anchor):
    cls = _CLS[r["risk"]]
    snippet = r["quote"] if len(r["quote"]) <= 700 else r["quote"][:700] + "…"
    hl = r.get("highlight", "")
    pos = snippet.lower().find(hl.lower()) if hl else -1
    if pos >= 0 and len(hl) < len(snippet):
        end = pos + len(hl)
        snip = escape(snippet[:pos]) + "<mark>" + escape(snippet[pos:end]) + "</mark>" + escape(snippet[end:])
    else:
        snip = escape(snippet) if snippet else '<span class="empty">no clean span located</span>'
    badge = "UNRECOGNIZED" if r["risk"] == "Unrecognized" else f"{r['risk'].upper()} RISK"
    pe = ""
    if r.get("plain_english"):
        warn = (f'<div class="conf">⚠️ The summarizer read this text as <b>{escape(r["summary_mismatch"])}</b>, '
                f'not {escape(r["category"])} — treat this line with caution.</div>' if r.get("summary_mismatch") else "")
        pe = f'<div class="pe"><span class="lbl2">In plain English · FLAN-T5</span>{escape(r["plain_english"])}</div>{warn}'
    jump = f'<a class="jump" href="#{anchor}">↧ Show in contract</a>' if anchor else ""
    maybe = '<span class="maybe">POSSIBLE — CHECK</span>' if r.get("possible") else ""
    return (
        f'<div class="clause {cls}{" possible" if r.get("possible") else ""}">'
        f'  <div class="top"><span class="name">{escape(r["title"])}</span>'
        f'    <span>{maybe}<span class="badge2 {cls}">{badge}</span></span></div>'
        f'  <div class="reason">{escape(r["risk_reason"])}</div>{pe}'
        f'  <div class="track"><div class="fill" style="width:{r["bar"]}%"></div></div>'
        f'  <div class="conf">{escape(r["bar_label"])}</div>'
        f'  <div class="snippet"><span class="lbl">{escape(r["snippet_label"])}</span>“{snip}”</div>{jump}'
        f'</div>'
    )


def render(res):
    rows, text = res["rows"], res["text"]
    n = {lvl: sum(1 for r in rows if r["risk"] == lvl) for lvl in LEVELS}
    m = {lvl: sum(1 for r in rows if r["risk"] == lvl and r.get("possible")) for lvl in LEVELS}
    of = lambda lvl: f' <span style="color:var(--muted);font-weight:600">({m[lvl]} possible)</span>' if m[lvl] else ""
    unk = (f'<span class="risk-pill"><span class="n pill-unk">⚪ {n["Unrecognized"]}</span> Unrecognized</span>'
           if n["Unrecognized"] else "")
    count = (f"🧩 {len(rows)} clause{'s' if len(rows) != 1 else ''}" if res["clause_mode"]
             else f"📄 {len(rows)} of 41 clause types")
    st.markdown(
        '<div class="risk-summary">'
        f'<span class="risk-pill"><span class="n pill-high">🔴 {n["High"]}</span> High risk{of("High")}</span>'
        f'<span class="risk-pill"><span class="n pill-med">🟠 {n["Medium"]}</span> Medium risk{of("Medium")}</span>'
        f'<span class="risk-pill"><span class="n pill-low">🟢 {n["Low"]}</span> Low risk{of("Low")}</span>'
        f'{unk}<span class="risk-pill">{count} · {escape(res["source"])}</span></div>',
        unsafe_allow_html=True)

    # ---- risk cards
    st.markdown('<div class="sec"><span class="dot blue"></span>Clauses, grouped by risk '
                f'<span style="font-weight:500;color:var(--muted);font-size:.85rem">— {escape(res["mode"])}</span></div>',
                unsafe_allow_html=True)
    anc = fx.anchors(text, rows)
    if not rows:
        st.markdown('<div class="glass"><span class="empty">No clause categories were detected '
                    'in this text.</span></div>', unsafe_allow_html=True)
    else:
        html = ""
        for lvl in LEVELS:
            group = [r for r in rows if r["risk"] == lvl]
            if group:
                head = "UNRECOGNIZED" if lvl == "Unrecognized" else f"{lvl.upper()} RISK"
                noun = "clause" if res["clause_mode"] else "clause type"
                sure = [r for r in group if not r.get("possible")]
                maybe = [r for r in group if r.get("possible")]
                html += (f'<div class="grp-head"><span class="tag {_CLS[lvl]}">{head}</span>'
                         f'<span class="count">{len(group)} {noun}{"s" if len(group) != 1 else ""}</span></div>'
                         + "".join(card(r, anc.get(id(r))) for r in sure))
                if maybe:
                    html += (f'<div class="grp-sub">Possible — check · {len(maybe)} with less than a 50% '
                             f'chance of being present</div>' + "".join(card(r, anc.get(id(r))) for r in maybe))
        st.markdown(f'<div class="glass">{html}</div>', unsafe_allow_html=True)
    if res.get("skipped"):
        with st.expander(f"{len(res['skipped'])} paragraph{'s' if len(res['skipped']) != 1 else ''} "
                         "not analyzed as clauses (title, preamble, signature block…)"):
            st.caption("Your text is made of numbered clauses, so only the numbered paragraphs are "
                       "analyzed. These un-numbered paragraphs were skipped:")
            for p in res["skipped"]:
                st.markdown(f"- {escape(p[:200])}{'…' if len(p) > 200 else ''}", unsafe_allow_html=True)
    if res["clause_mode"]:
        ev = mu.clause_eval()
        if ev:
            st.caption(
                f"When a clause heading names its category, the heading is used. Otherwise the model "
                f"decides: on {ev['n_test_clauses']} held-out CUAD clauses it picked the right one of 41 "
                f"categories {ev['calibrated_acc']:.0%} of the time (right one in its top 3: "
                f"{ev['calibrated_top3']:.0%}; right risk level: {ev['risk_level_acc']:.0%}).")
    if any(r.get("plain_english") for r in rows):
        st.caption("The plain-English lines come from the fine-tuned FLAN-T5 summarizer. As the project "
                   "report shows, it describes the clause's category in one sentence rather than "
                   "summarizing its exact wording.")

    # ---- highlighted contract
    st.markdown('<div class="sec"><span class="dot blue"></span>Your contract, highlighted by risk</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="contract-view">{fx.highlighted_html(text, rows)}</div>', unsafe_allow_html=True)

    # ---- missing-protection checklist
    st.markdown('<div class="sec"><span class="dot blue"></span>Missing-protection checklist</div>',
                unsafe_allow_html=True)
    found = {r["category"] for r in rows}
    items_html = ""
    for c, why, needs in mu.PROTECTIONS:
        if needs and needs not in found:
            continue
        ok = c in found
        items_html += (f'<div class="check"><span class="ic">{"✅" if ok else "⚠️"}</span><div>'
                       f'<div class="nm">{escape(c)} — {"found" if ok else "not detected"}</div>'
                       + ("" if ok else f'<div class="wy">{escape(why)}</div>') + '</div></div>')
    st.markdown(f'<div class="glass">{items_html}</div>', unsafe_allow_html=True)
    st.caption("“Not detected” means the models did not find it — they can miss clauses, so check the "
               "contract before relying on a missing item.")

    # ---- downloadable report
    st.markdown('<div class="sec"><span class="dot blue"></span>Download the report</div>',
                unsafe_allow_html=True)
    stem = "clausify_report"
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️  PDF report", fx.results_pdf(rows, res["missing"], res),
                           file_name=f"{stem}.pdf", mime="application/pdf",
                           use_container_width=True, on_click="ignore")
    with c2:
        st.download_button("⬇️  CSV (spreadsheet)", fx.results_csv(rows), file_name=f"{stem}.csv",
                           mime="text/csv", use_container_width=True, on_click="ignore")


if st.session_state.get("result"):
    res = st.session_state["result"]
    if (contract_text or "") != res["text"]:
        st.warning("⚠️ These results are for the **previously analyzed** text "
                   f"({escape(res['source'])}), not what is in the box now. "
                   "Click **Analyze contract** to analyze the current text.")
    render(res)
