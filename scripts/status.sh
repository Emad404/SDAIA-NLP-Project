#!/bin/bash
# concise status of all background logs
. /content/env.sh
for f in $BAYAN_WORK/logs/*.log; do
  echo "##### $(basename $f)"
  grep -E '^RESULT|^VAL|^\{|^==|Traceback|Error|DONE|clitic|by language' $f | cut -c1-230 | tail -${1:-14}
done
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
