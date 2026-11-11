#!/bin/bash
. /content/env.sh
cd /content/bayan_project
while pgrep -f run_s02.sh >/dev/null; do sleep 15; done
bash scripts/run_s03_s04.sh
