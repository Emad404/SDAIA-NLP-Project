#!/bin/bash
# case-augmented generalist -> pick teacher -> distilled students -> Gulf adaptation
. /content/env.sh
cd /content/bayan_project
while pgrep -f "queue_s06b.sh|queue_s02c.sh|s06b_distill|s02c_gulf" >/dev/null; do sleep 5; done
python scripts/s02_train_cls.py --task topic --name topic_xlmr_all_ca --ckpt xlm-roberta-base --lang all --profile xlmr --epochs 4 --lr 3e-5 --bs 32 --case_aug 0.35
T=$(python - <<'PY'
import json
a=json.load(open("/content/bayan_project/reports/train_topic_xlmr_all.json"))["val_macro_f1"]
b=json.load(open("/content/bayan_project/reports/train_topic_xlmr_all_ca.json"))["val_macro_f1"]
print("topic_xlmr_all_ca" if b >= a - 0.005 else "topic_xlmr_all")
PY
)
echo "TEACHER/BASE = $T"
python scripts/s06b_distill.py --teacher $T --layers 2,5,8,11 --name topic_xlmr_l4_kd
python scripts/s06b_distill.py --teacher $T --layers 3,7,11 --name topic_xlmr_l3_kd
python scripts/s02c_gulf_adapt.py --base $T
echo S02D ALL DONE
