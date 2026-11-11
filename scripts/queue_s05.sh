#!/bin/bash
. /content/env.sh
cd /content/bayan_project
while pgrep -f "run_s02.sh|run_s03_s04.sh|queue_s03.sh" >/dev/null; do sleep 15; done
python scripts/s05_search.py
echo S05 ALL DONE
