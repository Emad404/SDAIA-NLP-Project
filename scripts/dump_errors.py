"""Print validation errors of a classifier compactly, for hand-reading (usage: dump_errors.py MODEL TASK OFFSET LIMIT)."""
import sys, os
import pandas as pd
W = os.environ.get("BAYAN_WORK", "/content/bayan_work")
m, task, off, lim = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
d = pd.read_csv(f"{W}/artifacts/preds/val_{m}.csv")
e = d[d.y_true != d.pred].sort_values("feedback_id").reset_index(drop=True)
if off == 0:
    print(f"{m} {task}: {len(e)} errors of {len(d)} val rows")
for i, r in e.iloc[off:off + lim].iterrows():
    print(f"{i}|{r.language}/{r.dialect_region}|{r.y_true}>{r.pred}|{r.conf:.2f}|{r.text[:150]}")
