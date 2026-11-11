#!/bin/bash
. /content/env.sh
cd /content/bayan_project
while pgrep -f "run_s02.sh|run_s03_s04.sh|queue_s03.sh|queue_s05.sh" >/dev/null; do sleep 15; done
python scripts/s02_train_cls.py --task topic --name topic_xlmr_large_all --ckpt xlm-roberta-large --lang all --profile xlmr --epochs 4 --lr 1.5e-5 --bs 16
echo S02B ALL DONE
