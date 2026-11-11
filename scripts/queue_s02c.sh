#!/bin/bash
. /content/env.sh
cd /content/bayan_project
while pgrep -f "queue_s06b.sh|s06b_distill" >/dev/null; do sleep 10; done
python scripts/s02c_gulf_adapt.py
echo S02C ALL DONE
