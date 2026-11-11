#!/bin/bash
# HTTP benchmarks of the student candidates (router off: the router is not shipped), after the canary text fix
. /content/env.sh
cd /content/bayan_project
while pgrep -f "queue_final.sh|s07c_http_bench" >/dev/null; do sleep 3; done
for M in topic_xlmr_l4_kd topic_xlmr_l3_kd topic_xlmr_all_ca; do for P in int8 fp32; do
  python scripts/s07c_http_bench.py --precision $P --model $M --no-route --tag _$M --seconds 25; done; done
echo BENCH2_DONE
