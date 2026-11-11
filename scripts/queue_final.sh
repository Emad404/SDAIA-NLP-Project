#!/bin/bash
# selection -> behaviour suite -> optimisation ladders -> HTTP benchmarks (CPU benchmarks run with nothing else training)
. /content/env.sh
cd /content/bayan_project
while pgrep -f "queue_s02d.sh|s02c_gulf|s06b_distill|s02_train_cls" >/dev/null; do sleep 5; done
python scripts/s07_select.py && python scripts/s07b_behaviour_errors.py
for M in topic_xlmr_all_ca topic_xlmr_l4_kd topic_xlmr_l3_kd; do
  python scripts/s06_optimize.py --model $M
done
for M in topic_xlmr_all_ca topic_xlmr_l3_kd; do
  for P in fp32 int8; do
    python scripts/s07c_http_bench.py --precision $P --model $M --no-route --tag _$M --seconds 25
  done
done
echo FINAL_QUEUE_DONE
