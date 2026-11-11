"""One line per HTTP-benchmark run (all reports/http_bench_*.json)."""
import json, glob, os
R = os.environ.get("BAYAN_REPORTS", "/content/bayan_project/reports")
for p in sorted(glob.glob(f"{R}/http_bench_*.json")):
    d = json.load(open(p))
    name = os.path.basename(p)[len("http_bench_"):-5]
    for r in d.get("runs", [d]):
        print(f"{name:34s} think={r.get('mean_think_ms', 0):5.0f} rps={r['rps']:5.1f} p50={r['p50_ms']:7.1f} p95={r['p95_ms']:7.1f} p99={r['p99_ms']:7.1f} ok40={r['meets_p99_le_40ms']}")
