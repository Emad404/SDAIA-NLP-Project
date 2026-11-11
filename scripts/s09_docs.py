"""Stage 9: render RESULTS.md, BENCHMARKS.md and one model card per artefact from the JSON reports.
Numbers come ONLY from reports/*.json (never typed by hand). The 'Limitations' sections are hand-written (below) and are
the authors' judgement; the 'Measured' sections are generated."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from bayan import config as C
from bayan.preproc import PREPROC_VERSION

R = C.REPORTS_DIR
ROOT = Path(__file__).resolve().parents[1]
MC = ROOT / "model_cards"
MC.mkdir(exist_ok=True)


def J(name, default=None):
    p = R / name
    return json.load(open(p)) if p.exists() else default


def f3(x):
    return "n/a" if x is None else (f"{x:.3f}" if isinstance(x, (int, float)) else str(x))


def ci(x):
    return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]" if isinstance(x, list) and len(x) == 3 else f3(x)


def mark(ok):
    return "MET" if ok else "NOT MET"


fin, cfg = J("final_test_results.json", {}), J("serving_config.json", {})
sel, sent_sel, ner_sel = J("topic_selection_val.json", {}), J("sentiment_selection_val.json", {}), J("ner_selection_val.json", {})
beh = J("behaviour_val.json", {})
srch_cfg, srch_cmp = J("search_config.json", {}), J("search_dev_stage_comparison.json", {})
bench = {p.stem.replace("http_bench_", ""): json.load(open(p)) for p in R.glob("http_bench_*.json")}
opt = {p.stem.replace("optimize_", ""): json.load(open(p)) for p in R.glob("optimize_*.json")}
gen = (cfg.get("topic") or {}).get("general")

# ------------------------------------------------------------------ RESULTS.md
L = ["# Bayan: results (auto-generated from reports/*.json)", "",
     "Selection (models, thresholds, precision) was made on VALIDATION data only. The frozen test split was evaluated once by "
     "`scripts/s08_final.py` (lock file `final_run.lock`). Every test number carries a 95% bootstrap CI.", ""]
if fin:
    t = fin.get("topic", {})
    L += ["## Requirement scorecard (frozen test)", "", "| Requirement | Measured | Verdict |", "|---|---|---|"]
    ship = t.get("topic_shipped_system", {})
    for base in ("topic_tfidf_word", "topic_tfidf_char"):
        d = t.get(f"shipped_minus_{base}", {})
        if d:
            L.append(f"| Topic classifier beats {base} by >= 8 macro-F1 points | {d['diff_macro_f1_points']:+.2f} pts, CI {d['ci_points']} | {mark(d['meets_plus_8_points'])} (CI lower bound >= 8: {d['ci_lower_meets_plus_8']}) |")
    g = fin.get("gulf_slice_dialect_aware_vs_general")
    if g:
        L.append(f"| Dialect-aware variant wins on Gulf slice | diff {g['diff']:+.4f}, CI {g['ci']}, n={g['n']} | {mark(g['dialect_aware_wins'])} (significant at 95%: {g['significant_at_95']}) |")
    p = fin.get("pii", {})
    if p:
        L.append(f"| PII masking recall = 100% (60-case set) | recall {p['recall']} ({p['leaks']} leaks) | {mark(p['leaks'] == 0)} |")
    n = fin.get("ner", {})
    if n:
        cl = n.get("clitic_tests", {})
        L.append(f"| NER entity-F1 >= 0.80 | {n['entity_f1']} (ar {n['by_language'].get('ar')}, en {n['by_language'].get('en')}) | {mark(n['meets_0.80'])} |")
        L.append(f"| NER clitic tests | pass rates {', '.join(f'{k}:{v['pass_rate']}' for k, v in cl.items())} | min {min(v['pass_rate'] for v in cl.values()) if cl else 'n/a'} |")
    q = fin.get("qa_smoke20", {})
    if q:
        frac = q["correct_strict_EM"] / max(q["n"], 1)
        L.append(f"| QA smoke set >= 17/20 (strict exact match; the smoke file shipped with the data has {q['n']} items, not 20, so the bar is applied as >= 85%) | "
                 f"{q['correct_strict_EM']}/{q['n']} = {frac:.0%} (lenient F1>=0.5: {q['correct_lenient_F1>=0.5']}) | {'MET' if frac >= 17 / 20 else 'NOT MET'} (tiny set: treat as a smoke test only) |")
    s = fin.get("search", {})
    if s:
        a = s.get("two_stage_anchor_rescore", {})
        L.append(f"| Search recall@10 >= 0.80 | {ci(a.get('recall@10'))} | {mark(s.get('meets_recall@10>=0.80'))} |")
        L.append(f"| Search MRR@10 >= 0.70 | {ci(a.get('mrr@10'))} | {mark(s.get('meets_mrr@10>=0.70'))} |")
    FINAL = [k for k in ("int8final_int8", "fp32final_fp32") if k in bench]
    for k in FINAL:
        for r in bench[k].get("runs", [bench[k]]):
            load = "saturation (no think time)" if not r.get("mean_think_ms") else f"mean think {r['mean_think_ms']:.0f} ms/user"
            L.append(f"| HTTP p99 <= 40 ms, 16 users, {k.split('final_')[1]} [{load}] | p99 {r['p99_ms']} ms, p50 {r['p50_ms']} ms, {r['rps']} rps | {mark(r['meets_p99_le_40ms'])} |")
    L += ["", "## Topic classifier (test)", "", "| System | macro-F1 [95% CI] | accuracy |", "|---|---|---|"]
    for k, v in t.items():
        if isinstance(v, dict) and "macro_f1" in v:
            L.append(f"| {k} | {v['macro_f1']} {v['ci']} | {v.get('accuracy')} |")
    L += ["", "## Sentiment (test)", "", "| System | macro-F1 [95% CI] |", "|---|---|"]
    for k, v in fin.get("sentiment", {}).items():
        if isinstance(v, dict) and "macro_f1" in v:
            L.append(f"| {k} | {v['macro_f1']} {v['ci']} |")
    for nm in ("slices_topic_test.csv", "per_class_topic_test.csv", "slices_sentiment_test.csv"):
        if (R / nm).exists():
            L += ["", f"### {nm.replace('.csv', '').replace('_', ' ')}", "", pd.read_csv(R / nm).round(4).to_markdown(index=False)]
else:
    L.append("_final test not run yet_")
if beh:
    L += ["", "## Behavioural suite (validation)", "", "| Suite | pass rate |", "|---|---|"]
    for k, v in beh.items():
        if isinstance(v, dict) and "pass_rate" in v:
            L.append(f"| {k} | {v['pass_rate']:.3f} (n={v.get('n')}) |")
        elif isinstance(v, dict) and "_mean" in v:
            L.append(f"| {k} invariance mean / min | {v['_mean']:.3f} / {v['_min']:.3f} |")
(ROOT / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")

# ------------------------------------------------------------------ BENCHMARKS.md
B = ["# Bayan: serving benchmarks (auto-generated)", ""]
for m, o in opt.items():
    B += [f"## Optimisation ladder: `{m}`", "", f"Hardware: {o.get('hardware')}", "", "| Rung | val macro-F1 | accuracy tax vs R1 | p50 ms | p95 ms | p99 ms |", "|---|---|---|---|---|---|"]
    for rn, r in o["rungs"].items():
        B.append(f"| {rn} | {r['val_macro_f1']} | {r['accuracy_tax_vs_R1']:+} | {r.get('p50')} | {r.get('p95')} | {r.get('p99')} |")
    B += ["", f"Sizes (MB): {o.get('sizes_mb')}  |  parity: {o.get('parity')}", "", f"Notes: {o.get('benchmark_notes')}", ""]
def _rows(k, v):
    out = []
    for r in v.get("runs", [v]):
        load = "none (saturation)" if not r.get("mean_think_ms") else f"{r['mean_think_ms']:.0f} ms"
        out.append(f"| {k} | {load} | {r['requests']} | {r['errors']} | {r['rps']} | {r['p50_ms']} | {r['p95_ms']} | {r['p99_ms']} | {r['max_ms']} | {r['meets_p99_le_40ms']} |")
    return out


HDR = ["| run | think time / user | requests | errors | rps | p50 | p95 | p99 | max | p99<=40ms |", "|---|---|---|---|---|---|---|---|---|---|"]
if bench:
    FINAL = [k for k in ("int8final_int8", "fp32final_fp32") if k in bench]
    B += ["## HTTP benchmark: shipped configuration (POST /v1/classify, 16 concurrent users)", "",
          "Server pinned to CPU 0 and the load generator to CPU 1, 1 ONNX thread, no micro-batch wait. `int8` is the default precision, `fp32` is the rollback. "
          "'Think time' is the mean exponential pause each simulated user takes between requests (0 = closed loop at saturation).", ""] + HDR
    for k in FINAL:
        B += _rows(k, bench[k])
    B += ["", "## HTTP benchmark: tuning history (not the shipped configuration)", "",
          "Earlier runs used 2 ONNX threads, a 2 ms batching wait and an unpinned load generator; they explain why the final settings were chosen. "
          "Saturation numbers of unpinned runs are contaminated by the load generator competing for the same 2 vCPUs.", ""] + HDR
    for k, v in bench.items():
        if k not in FINAL:
            B += _rows(k, v)
    any_ = next(iter(bench.values()))
    B += ["", f"Caveat: {any_.get('caveat')}", f"Hardware: {any_.get('hardware')}"]
(ROOT / "BENCHMARKS.md").write_text("\n".join(B) + "\n", encoding="utf-8")

# ------------------------------------------------------------------ model cards (limitations are hand-written)
COMMON = ("Data are the course's synthetic Bayan corpora (templated messages, synthetic dialect labels, a few percent label noise). "
          "No number in this card has been validated on real citizen traffic.")
CARDS = {
    "topic_classifier": ("Topic classifier (generalist)", gen, "Routes a message to one of 8 service topics (appointments, billing, waste, permits, outage, account_access, digital_services, transport).",
        ["The corpus is templated and short (3-39 words); accuracy on long, multi-issue or code-switched messages is untested.",
         "Messages mentioning two topics are labelled with the first one only; the model cannot return a second topic.",
         "It leans on keywords: in our hand-written minimum-functionality set it labelled 'My electricity bill is much higher than usual' as outage and 'I need a permit to renovate my house' as waste.",
         "About 3-4% of training labels are noisy by construction, so the ceiling on this data is below 100% and small differences between models are inside noise.",
         "Gulf-dialect behaviour is measured against SYNTHETIC dialect labels; it says little about real Gulf speech.",
         "Upper-cased English input was the weakest invariance check on the first model (about 80% unchanged predictions); the shipped model is trained with case augmentation, but other surface changes (typos, emoji, transliterated Arabic) were not tested."]),
    "dialect_aware_classifier": ("Dialect-aware topic classifier (Gulf variant) + dialect router", (cfg.get("topic") or {}).get("dialect_aware"),
        "Optional second classifier used only for Arabic messages the dialect router predicts as Gulf.",
        ["The router itself is a char-n-gram linear model with roughly 77% validation accuracy over four dialects, so a fifth or more of Arabic messages may go to the wrong branch.",
         "Whether this variant beats the generalist is a statistical question with small samples; read the paired CI in RESULTS.md, not only the point estimate.",
         "If the CI includes zero the honest conclusion is 'no detectable difference', and routing adds complexity without proven benefit."]),
    "sentiment_classifier": ("Sentiment classifier", cfg.get("sentiment"), "Negative / neutral / positive.",
        ["About 8% of training labels are noisy.",
         "On our hand-written minimum-functionality cases the model labelled explicit emotion statements such as 'I am angry that nobody answered my calls' as neutral: it seems to key on dataset style (a described problem) rather than on emotion words.",
         "Sarcasm, negation and intensity were not systematically tested beyond a few dozen hand-written cases (see behavioural suite)."]),
    "ner_tagger": ("Entity extractor (PERSON, LOCATION, SERVICE)", cfg.get("ner"), "BIO token tagger returning entity spans with character offsets.",
        ["Only three entity types exist; organisations, dates, amounts and IDs are not extracted (PII is masked by the preprocessing module instead).",
         "The dataset is highly templated, so entity-F1 near 1.0 mostly shows the model learned the templates; do not expect that on free text.",
         "Arabic clitic robustness (و ب ل ف وب ول فب prefixes) was tested with synthetic prefixing of gold entities; real morphology is richer than that test."]),
    "qa_model": ("Extractive QA with honest nulls", cfg.get("qa"), "Answers a question from a supplied service document or abstains (null answer) when the document does not contain it.",
        ["The abstain threshold is tuned on 64 validation questions; at that sample size it is imprecise.",
         "Questions are answered from the single context passed in; the service does not retrieve documents for the user (the /v1/ask endpoint takes a doc_id).",
         "Contexts are 3-6 short facts; long documents would need chunking that is not implemented."]),
    "retrieval": ("Similar-case retrieval (bi-encoder + FAISS + stage-2 anchor re-scorer)", srch_cfg.get("embedder"),
        "Returns the most similar historical cases (cross-language) or an explicit no-answer.",
        ["Stage 2 exploits a property of THIS corpus: every case ends with a district and a date, and queries often repeat them. On a corpus without such anchors, stage 2 does nothing and quality falls back to the dense retriever.",
         "The stage-2 weights were tuned on 65 development queries; the test queries (75) were scored once.",
         "The district lexicon covers the 17 districts in the case base; a query naming an unknown district gets no district bonus.",
         "Off-the-shelf cross-encoder re-rankers (mMARCO mMiniLM, bge-reranker-base) did NOT beat the dense retriever on development queries and are therefore not shipped.",
         "The no-answer threshold separates 'out of domain' queries well but was tuned on only about 15 dev no-answer queries."]),
}
for key, (title, art, use, lim) in CARDS.items():
    body = [f"# Model card: {title}", "", f"- Artefact: `{art}`", f"- Intended use: {use}", f"- Preprocessing: `bayan.preproc` v{PREPROC_VERSION} (single shared module; the model owns its Arabic profile)",
            "- Data: " + COMMON, "", "## Measured", ""]
    if key == "topic_classifier":
        t = fin.get("topic", {})
        body += [f"- Test macro-F1 (shipped system): {t.get('topic_shipped_system')}", f"- fp32 / int8 test macro-F1: {t.get('topic_general_ort_fp32')} / {t.get('topic_general_ort_int8')}",
                 f"- Validation table: see `reports/topic_selection_val.json`"]
        for m, o in opt.items():
            if m == gen:
                body += ["- Optimisation ladder: see BENCHMARKS.md", f"- Quantisation decision: {cfg.get('topic', {}).get('quantisation_decision', 'see DECISIONS.md')}"]
    elif key == "dialect_aware_classifier":
        body += [f"- Gulf slice paired comparison (test): {fin.get('gulf_slice_dialect_aware_vs_general')}", f"- Validation comparison: {sel.get('selection')}"]
    elif key == "sentiment_classifier":
        body += [f"- Test: {fin.get('sentiment')}", f"- Validation: {sent_sel}"]
    elif key == "ner_tagger":
        body += [f"- Test: {fin.get('ner')}", f"- Validation choice: {ner_sel}"]
    elif key == "qa_model":
        body += [f"- Test split: {fin.get('qa_test_split')}", f"- Smoke 20: { {k: v for k, v in fin.get('qa_smoke20', {}).items() if k != 'rows'} }"]
    else:
        body += [f"- Test: {fin.get('search')}", f"- Dev stage comparison: {srch_cmp}", f"- Config: {srch_cfg}"]
    body += ["", "## Limitations (hand-written)", ""] + [f"- {x}" for x in lim]
    body += ["", "## Rollback", "", "Serve with `BAYAN_PRECISION=fp32` (default) for the unquantised reference; skew canaries are asserted at start-up and the service refuses to start if they fail.", ""]
    (MC / f"{key}.md").write_text("\n".join(body), encoding="utf-8")
print("DOCS RENDERED:", [p.name for p in MC.glob("*.md")])
