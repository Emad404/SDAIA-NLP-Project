"""Compact dump of the numbers the README/DECISIONS quote (reads reports/*.json)."""
import json, os
from pathlib import Path
R = Path(os.environ.get("BAYAN_REPORTS", "/content/bayan_project/reports"))
def J(n):
    p = R / n
    return json.load(open(p)) if p.exists() else {}
def short(o, n=700):
    return json.dumps(o, ensure_ascii=False)[:n]
sel = J("topic_selection_val.json")
tb = sel.get("table") or {}
print("TOPIC SELECTION table:"); [print("  ", k, short(v, 260)) for k, v in (tb.items() if isinstance(tb, dict) else enumerate(tb))]
print("paired vs generalist:", short(sel.get("paired_vs_generalist"), 900))
print("selection:", short(sel.get("selection"), 600))
print("SERVING CFG:", short(J("serving_config.json"), 900))
print("QUANT:", short({k: (v.get("decision"), v.get("why")) for k, v in J("quant_decisions.json").get("models", {}).items()}, 900))
f = J("final_test_results.json")
print("FINAL keys:", list(f))
for k in ("pii", "ner", "qa_test_split", "qa", "search"):
    print(k, short(f.get(k), 900))
print("BEH:", short({k: (v if not isinstance(v, dict) else {a: b for a, b in v.items() if a != "failures"}) for k, v in J("behaviour_val.json").items()}, 1200))
print("SEARCH CFG:", short(J("search_config.json"), 700))
print("SENT SEL:", short(J("sentiment_selection_val.json"), 500))
print("NER SEL:", short(J("ner_selection_val.json"), 500))
ea = J("error_analysis_topic_val.json")
print("ERR keys", list(ea)); print("ERR:", short({k: v for k, v in ea.items() if k not in ("rows", "errors")}, 1500))
