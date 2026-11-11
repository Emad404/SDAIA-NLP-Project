#!/bin/bash
. /content/env.sh
cd /content/bayan_project
while pgrep -f "queue_s02b.sh|queue_s03b.sh|s03_train_ner|s02_train_cls" >/dev/null; do sleep 10; done
python scripts/s06b_distill.py --layers 2,5,8,11 --name topic_xlmr_l4_kd
python scripts/s06b_distill.py --layers 3,7,11 --name topic_xlmr_l3_kd
echo S06B ALL DONE
