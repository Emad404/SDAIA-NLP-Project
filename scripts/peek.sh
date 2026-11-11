#!/bin/bash
# usage: bash scripts/peek.sh <logname> [N]   (key lines of /content/bayan_work/logs/<logname>.log)
L=/content/bayan_work/logs/$1.log
N=${2:-12}
grep -a -E 'RESULT|TEACHER|teacher val|epoch|early|Traceback|Error|DONE|VAL |decision|STAGE' "$L" | cut -c1-230 | tail -n "$N"
echo "--- running:"; pgrep -af 'python3? scripts' | cut -c1-120
