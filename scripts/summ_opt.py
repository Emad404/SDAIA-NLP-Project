"""Compact print of the optimisation ladders (reports/optimize_*.json)."""
import json, glob, os
R = os.environ.get("BAYAN_REPORTS", "/content/bayan_project/reports")
for p in sorted(glob.glob(f"{R}/optimize_*.json")):
    d = json.load(open(p))
    print("##", os.path.basename(p)[9:-5], "sizes", d.get("sizes_mb"), "parity", d.get("parity"))
    for k, v in d["rungs"].items():
        print(f"  {k:26s} f1={v['val_macro_f1']:.4f} p50={v['p50']:7.1f} p95={v['p95']:7.1f} p99={v['p99']:7.1f}  tax_vs_ref={v.get('accuracy_tax_vs_R1')}")
