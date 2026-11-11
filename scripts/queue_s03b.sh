#!/bin/bash
. /content/env.sh
cd /content/bayan_project
while pgrep -f "run_s02.sh|run_s03_s04.sh|queue_s03.sh|queue_s05.sh|queue_s02b.sh" >/dev/null; do sleep 15; done
python scripts/s03_train_ner.py --name ner_xlmr_clitic_aug --clitic_aug 0.35
echo S03B ALL DONE
