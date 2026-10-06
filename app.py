"""
Clausify — premium Streamlit demo (CUAD contract analysis).

Add a contract (upload a .txt file OR paste the text) and see BOTH trained
models work on it, live:

  Model 1 — Presence Classifier : which of the 41 clause categories are present
  Model 2 — Span Extractor      : the exact clause text pulled from the contract

Run:  streamlit run app.py
"""

from html import escape

import streamlit as st

import model_utils as mu

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
.clause .badge2{ font-size:.72rem; font-weight:800; letter-spacing:.06em;
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
    "for the party with less bargaining power, and have not been reviewed by a lawyer. The risk "
    "badge and category are more reliable than the quoted text, which may not be the exact clause "
    "— read each card as \"look here\". Scores and bars are model scores, not probabilities. "
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
# Input — two "bars" to add a contract
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
    uploaded = st.file_uploader("Upload a contract file (.txt)", type=["txt"])
    if uploaded is not None:
        contract_text = uploaded.read().decode("utf-8", errors="ignore")
        source = f"uploaded file · {uploaded.name}"

analyze_as = st.radio(
    "Analyze as",
    ["✨ Auto", "📄 Whole contract", "🧩 Separate clauses"],
    horizontal=True,
    help="Separate clauses: every paragraph you paste is one clause and gets exactly one "
         "category and risk level. Whole contract: the models scan the document for all 41 "
         "clause types. Auto picks Separate clauses for a short list of paragraphs.",
)
threshold = 0.5   # fixed decision threshold (the default used in every reported evaluation)
go = st.button("🔍  Analyze contract", type="primary", disabled=not contract_text)

# --------------------------------------------------------------------------- #
# Result renderer — clauses grouped by risk (High / Medium / Low), poster-style
# --------------------------------------------------------------------------- #
_CLS = {"High": "high", "Medium": "med", "Low": "low", "Unrecognized": "unk"}
LEVELS = ("High", "Medium", "Low", "Unrecognized")


def card(title, risk, reason, bar_pct, bar_label, snippet_label, snippet, highlight=""):
    cls = _CLS[risk]
    if not snippet:
        snip = '<span class="empty">no clean span located</span>'
    else:
        pos = snippet.lower().find(highlight.lower()) if highlight else -1
        if pos >= 0 and len(highlight) < len(snippet):
            end = pos + len(highlight)
            snip = (escape(snippet[:pos]) + "<mark>" + escape(snippet[pos:end]) + "</mark>"
                    + escape(snippet[end:]))
        else:
            snip = escape(snippet)
    badge = "UNRECOGNIZED" if risk == "Unrecognized" else f"{risk.upper()} RISK"
    return (
        f'<div class="clause {cls}">'
        f'  <div class="top"><span class="name">{escape(title)}</span>'
        f'    <span class="badge2 {cls}">{badge}</span></div>'
        f'  <div class="reason">{escape(reason)}</div>'
        f'  <div class="track"><div class="fill" style="width:{bar_pct}%"></div></div>'
        f'  <div class="conf">{escape(bar_label)}</div>'
        f'  <div class="snippet"><span class="lbl">{escape(snippet_label)}</span>“{snip}”</div>'
        f'</div>'
    )


def doc_card(it):
    pct = round(it["presence_score"] * 100)
    text = it["span_text"] if len(it["span_text"]) <= 700 else it["span_text"][:700] + "…"
    return card(it["category"], it["risk"], it["reason"], pct,
                f"Model 1 · presence score {pct}%",
                "Located clause · Model 1 picks the paragraph, Model 2 highlights the key span",
                text, it.get("highlight", ""))


def clause_mode_card(it):
    z = it["z"]
    strength = "strong" if z >= 4 else "moderate" if z >= 2.5 else "weak"
    pct = int(max(5, min(100, z / 6 * 100)))
    title = f'Clause {it["index"]} · {it["category"] or "No clear category"}'
    if it["category"] and it["source"] == "heading":
        pct = 100
        agree = "model agrees" if it["model_guess"] == it["category"] else f"model's own guess: {it['model_guess']}"
        label = f"Category from the clause heading ({agree})"
    else:
        label = None
    label = label or (f"Category match (model): {strength} (score {z:.1f}; runner-up: {it['runner_up']})"
             if it["category"] else f"Closest category: {it['best_guess']} — match too weak to assign (score {z:.1f})")
    text = it["clause"] if len(it["clause"]) <= 600 else it["clause"][:600] + "…"
    return card(title, it["risk"], it["reason"], pct, label, "Your clause", text)


def risk_group(items, level, render, noun):
    group = [it for it in items if it["risk"] == level]
    if not group:
        return ""
    cls = _CLS[level]
    head = "UNRECOGNIZED" if level == "Unrecognized" else f"{level.upper()} RISK"
    return (
        f'<div class="grp-head"><span class="tag {cls}">{head}</span>'
        f'<span class="count">{len(group)} {noun}{"s" if len(group) != 1 else ""}</span></div>'
        + "".join(render(it) for it in group)
    )


def summary(items, last_pill):
    n = {lvl: sum(1 for it in items if it["risk"] == lvl) for lvl in LEVELS}
    unk = (f'<span class="risk-pill"><span class="n pill-unk">⚪ {n["Unrecognized"]}</span> Unrecognized</span>'
           if n["Unrecognized"] else "")
    st.markdown(
        '<div class="risk-summary">'
        f'<span class="risk-pill"><span class="n pill-high">🔴 {n["High"]}</span> High risk</span>'
        f'<span class="risk-pill"><span class="n pill-med">🟠 {n["Medium"]}</span> Medium risk</span>'
        f'<span class="risk-pill"><span class="n pill-low">🟢 {n["Low"]}</span> Low risk</span>'
        f'{unk}<span class="risk-pill">{last_pill}</span>'
        '</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
if go and contract_text:
    use_clauses = mu.clause_mode_available() and (
        analyze_as == "🧩 Separate clauses"
        or (analyze_as == "✨ Auto" and mu.looks_like_clause_list(contract_text)))
    clauses = mu.split_clauses(contract_text) if use_clauses else []
    if use_clauses and not clauses:
        st.warning("No clauses of at least 40 characters were found; analyzing as a whole contract.")
        use_clauses = False

    prog = st.progress(0.0, text="Model 1 reading…")

    def _update(frac):
        prog.progress(frac, text=f"Model 1 reading… {frac*100:.0f}%")

    if use_clauses:
        with st.spinner("Loading models & classifying each clause…"):
            mu.warm_up()
            items = mu.classify_clauses(clauses, progress=_update)
        prog.empty()
        summary(items, f"🧩 {len(items)} clause{'s' if len(items) != 1 else ''} · {escape(source)}")
        st.markdown('<div class="sec"><span class="dot blue"></span>Your clauses, grouped by risk '
                    '<span style="font-weight:500;color:var(--muted);font-size:.85rem">'
                    '— one category per clause</span></div>', unsafe_allow_html=True)
        html = "".join(risk_group(items, lvl, clause_mode_card, "clause") for lvl in LEVELS)
        st.markdown(f'<div class="glass">{html}</div>', unsafe_allow_html=True)
        ev = mu.clause_eval()
        if ev:
            st.caption(
                f"When a clause heading names its category, the heading is used. Otherwise the model "
                f"decides: on {ev['n_test_clauses']} held-out CUAD clauses it picked the right one of 41 "
                f"categories {ev['calibrated_acc']:.0%} of the time (right one in its top 3: "
                f"{ev['calibrated_top3']:.0%}; right risk level: {ev['risk_level_acc']:.0%}). "
                f"Check model-decided cards against the clause text.")
        with st.expander("See every clause with its category and runner-up"):
            st.dataframe(
                {"#": [it["index"] for it in items],
                 "category": [it["category"] or "—" for it in items],
                 "decided by": [it["source"] for it in items],
                 "model's guess": [it["model_guess"] for it in items],
                 "risk": [it["risk"] for it in items],
                 "match score": [round(it["z"], 2) for it in items],
                 "runner-up": [it["runner_up"] for it in items],
                 "clause": [it["clause"][:120] for it in items]},
                use_container_width=True, hide_index=True,
            )
    else:
        with st.spinner("Loading models & running inference…"):
            mu.warm_up()
            present, all_scores = mu.analyze(contract_text, threshold=threshold, progress=_update)
        prog.empty()
        summary(present, f"📄 {len(present)} of 41 clause types · {escape(source)}")
        st.markdown('<div class="sec"><span class="dot blue"></span>Risk-prioritized clauses '
                    '<span style="font-weight:500;color:var(--muted);font-size:.85rem">'
                    '— Model 1 detects, Model 2 extracts</span></div>', unsafe_allow_html=True)
        if not present:
            st.markdown('<div class="glass"><span class="empty">No clause categories were '
                        'detected in this text.</span></div>', unsafe_allow_html=True)
        else:
            html = "".join(risk_group(present, lvl, doc_card, "clause type") for lvl in LEVELS[:3])
            st.markdown(f'<div class="glass">{html}</div>', unsafe_allow_html=True)
        with st.expander("See all 41 categories, risk level, and presence scores"):
            rows = sorted(all_scores.items(),
                          key=lambda kv: (mu.RISK_ORDER[mu.risk_of(kv[0])[0]], -kv[1]))
            st.dataframe(
                {"category": [c for c, _ in rows],
                 "risk": [mu.risk_of(c)[0] for c, _ in rows],
                 "presence score": [round(s, 3) for _, s in rows],
                 "present?": ["✅" if s >= threshold else "" for _, s in rows]},
                use_container_width=True, hide_index=True,
            )
