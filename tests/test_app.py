"""
End-to-end checks for the Clausify app (Streamlit AppTest + direct module tests).

    cd "final project app p3"
    python3 -m pytest tests/test_app.py -q          # or: python3 tests/test_app.py

Needs the trained models (local or downloadable). Takes a few minutes on CPU.
"""
import io, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(HERE)
sys.path.insert(0, APP_DIR)
os.chdir(APP_DIR)

import model_utils as mu          # noqa: E402
import features as fx             # noqa: E402
from streamlit.testing.v1 import AppTest   # noqa: E402

NINE = open(os.path.join(HERE, "nine_clauses.txt")).read()
DEMO = open(os.path.join(APP_DIR, "test_contract.txt")).read()


def run_app(text=None, analyze_as=None, rule=None, summaries=None, upload=None):
    at = AppTest.from_file("app.py", default_timeout=900).run()
    if text is not None:
        at.text_area[0].input(text).run()
    if analyze_as:
        at.radio[1].set_value(analyze_as).run()
    if rule:
        at.radio[2].set_value(rule).run()
    if summaries is not None:
        at.checkbox[0].set_value(summaries).run()
    at.button[0].click().run()
    return at


def page(at):
    md = " ".join(m.value for m in at.markdown)
    return {
        "exceptions": [str(e.value)[:200] for e in at.exception],
        "errors": [e.value for e in at.error],
        "warnings": [w.value for w in at.warning],
        "pills": dict((k, int(v)) for k, v in re.findall(r'(🔴|🟠|🟢|⚪) (\d+)</span>', md)),
        "cards": md.count('class="clause '),
        "plain": md.count("In plain English"),
        "links": len(re.findall(r'href="#(clause-\d+)"', md)),
        "anchors": len(re.findall(r'<mark id="(clause-\d+)"', md)),
        "link_targets_ok": set(re.findall(r'href="#(clause-\d+)"', md)) <= set(re.findall(r'<mark id="(clause-\d+)"', md)),
        "raw_dollar": bool(re.search(r'\$\d', md)),            # unescaped $ (would be maths)
        "downloads": [b.label for b in at.get("download_button")],
        "md": md,
    }


results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")


# 1. clause mode, auto-detected -------------------------------------------------------
t = time.time(); p = page(run_app(NINE))
check("clause mode: no exceptions", not p["exceptions"], p["exceptions"])
check("clause mode: 9 cards, 3/3/3", p["cards"] == 9 and p["pills"] == {"🔴": 3, "🟠": 3, "🟢": 3}, f'{p["pills"]} {time.time()-t:.0f}s')
check("clause mode: plain-English line per card", p["plain"] == 9, p["plain"])
check("clause mode: every card links to a highlight", p["links"] == 9 and p["link_targets_ok"], (p["links"], p["anchors"]))
check("clause mode: $ escaped", not p["raw_dollar"])
check("clause mode: PDF + CSV downloads", len(p["downloads"]) == 2, p["downloads"])

# 2. whole contract, default (DistilBERT) and ensemble, summaries off ------------------
p = page(run_app(DEMO, analyze_as="📄 Whole contract"))
check("whole contract: no exceptions", not p["exceptions"], p["exceptions"])
check("whole contract: 30 types, 7 High", sum(p["pills"].values()) == 30 and p["pills"].get("🔴") == 7, p["pills"])
check("whole contract: links resolve to highlights", p["link_targets_ok"], (p["links"], p["anchors"]))
check("whole contract: $ escaped", not p["raw_dollar"])
p = page(run_app(DEMO, analyze_as="📄 Whole contract", rule="AND", summaries=False))
check("AND-ensemble: runs", not p["exceptions"], p["exceptions"])
check("AND-ensemble: fewer detections than DistilBERT", 0 < sum(p["pills"].values()) < 30, p["pills"])
check("summaries off: no plain-English lines", p["plain"] == 0, p["plain"])

# 3. auto mode on a full contract picks whole-contract mode --------------------------
p = page(run_app(DEMO))
check("auto on full contract -> whole-contract mode", "whole contract" in p["md"], "")

# 3b. stale results are flagged when the input changes after an analysis ---------------
at = run_app(NINE); at.text_area[0].input("A different contract text entirely, not analyzed yet.").run()
check("stale results flagged after editing input", any("previously analyzed" in w.value for w in at.warning),
      [w.value[:60] for w in at.warning])
at.text_area[0].input(NINE).run()
check("no stale warning when input matches", not any("previously analyzed" in w.value for w in at.warning))

# 4. edge inputs -----------------------------------------------------------------------
p = page(run_app("Short text."))
check("tiny input: no crash", not p["exceptions"], p["exceptions"])
p = page(run_app("Short text one here.", analyze_as="🧩 Separate clauses"))
check("tiny input in clause mode: warns and falls back", not p["exceptions"] and p["warnings"], p["warnings"])
weird = ("1. GOVERNING LAW. <script>alert(1)</script> This Agreement is governed by the laws of "
         "New York & Delaware; fees of $5,000 apply. ✓ 中文 *bold* _x_ `code`\n\n"
         "2. INSURANCE. Supplier shall carry insurance of $1,000,000 per occurrence at all times.")
p = page(run_app(weird))
check("special chars: no crash", not p["exceptions"], p["exceptions"])
check("special chars: HTML escaped", "<script>" not in p["md"] and "&lt;script&gt;" in p["md"])
check("special chars: $ escaped", not p["raw_dollar"])
no_blank = "\n".join(l for l in NINE.split("\n") if l.strip())          # numbered lines, no blank lines
p = page(run_app(no_blank, analyze_as="🧩 Separate clauses"))
check("numbered lines without blank lines: 9 clauses", p["cards"] == 9, p["cards"])

# 5. long contract (CUAD test contract > 30 windows) -----------------------------------
cuad = os.path.expanduser("~/p3_notebook/CUADv1.json")
if os.path.exists(cuad):
    js = json.load(open(cuad))["data"]
    longest = max(js, key=lambda c: len(c["paragraphs"][0]["context"]))["paragraphs"][0]["context"]
    t = time.time(); p = page(run_app(longest, analyze_as="📄 Whole contract", summaries=False))
    check("longest CUAD contract: no crash", not p["exceptions"], f"{len(longest):,} chars, {time.time()-t:.0f}s, {p['pills']}")

# 6. file uploads (direct extraction) --------------------------------------------------
import fitz, docx
d = docx.Document(); [d.add_paragraph(x) for x in DEMO.split("\n\n")]; b = io.BytesIO(); d.save(b)
txt, note = fx.extract_text("c.docx", b.getvalue())
check("docx extraction", len(txt) > 4000 and not note, len(txt))
doc = fitz.open(); pg = doc.new_page(); pg.insert_textbox(fitz.Rect(40, 40, 560, 800), DEMO[:3000], fontsize=7)
txt, note = fx.extract_text("c.pdf", doc.tobytes())
check("pdf extraction", len(txt) > 2000 and not note, len(txt))
blank = fitz.open(); blank.new_page()
txt, note = fx.extract_text("scan.pdf", blank.tobytes())
check("image-only pdf: warns", bool(note), note[:60])
try:
    fx.extract_text("broken.pdf", b"not a pdf"); ok = False
except Exception:
    ok = True
check("corrupt pdf: raises (app shows error)", ok)
txt, _ = fx.extract_text("win.txt", "Café clause — £100".encode("cp1252"))
check("Windows (cp1252) txt keeps accents", txt == "Café clause — £100", repr(txt))
txt, _ = fx.extract_text("bom.txt", "\ufeffGoverning law clause".encode("utf-8"))
check("UTF-8 BOM stripped", txt == "Governing law clause", repr(txt))

# 7. report generation edge cases -------------------------------------------------------
pdf = fx.results_pdf([], mu.missing_protections([]), {"source": "x", "mode": "y"})
check("PDF report with no clauses", pdf[:4] == b"%PDF", len(pdf))
row = {"#": 1, "category": "Governing Law", "risk": "Low", "risk_reason": "r", "quoted_text": "<b>$5 & ✓ 中文</b>",
       "plain_english": "p", "summary_mismatch": "Insurance", "score_text": "s"}
pdf = fx.results_pdf([row], [], {"source": "x", "mode": "y"})
check("PDF report with special chars", pdf[:4] == b"%PDF", len(pdf))
csv_bytes = fx.results_csv([row]); csv_text = csv_bytes.decode("utf-8-sig")
check("CSV has Excel BOM and keeps unicode", csv_bytes[:3] == b"\xef\xbb\xbf" and "中文" in csv_text and csv_text.count("\n") == 2, csv_text[:80])

# 8. checklist logic ---------------------------------------------------------------------
m = {x["category"] for x in mu.missing_protections(["Renewal Term"])}
check("checklist: renewal without notice period is flagged", "Notice Period To Terminate Renewal" in m)
m = {x["category"] for x in mu.missing_protections([])}
check("checklist: notice period not flagged without renewal", "Notice Period To Terminate Renewal" not in m)

failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
if __name__ == "__main__":
    sys.exit(1 if failed else 0)


def test_all():           # pytest entry point
    assert not failed, failed
