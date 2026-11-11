#!/bin/bash
# HTTP benchmarks (router off) for every serving candidate: closed-loop saturation (think 0) + paced users (think 100/250/500 ms mean)
. /content/env.sh
cd /content/bayan_project
for M in topic_xlmr_l4_kd topic_xlmr_l3_kd topic_xlmr_all_ca; do
  python scripts/s07c_http_bench.py --precision int8 --model $M --no-route --tag _$M --seconds 20 --think-ms-list 0,100,250,500
done
for M in topic_xlmr_l4_kd topic_xlmr_all_ca; do
  python scripts/s07c_http_bench.py --precision fp32 --model $M --no-route --tag _$M --seconds 20 --think-ms-list 0,250
done
echo BENCH3_DONE
