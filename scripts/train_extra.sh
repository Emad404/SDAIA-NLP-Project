#!/bin/bash
# Second training wave (everything that was added after the first model comparison). Portable: needs only PYTHONPATH/BAYAN_WORK.
set -e
P="python scripts/s02_train_cls.py"
python scripts/s03_train_ner.py --name ner_xlmr_clitic_aug --clitic_aug 0.35                       # NER with clitic augmentation
$P --task topic --name topic_xlmr_large_all --ckpt xlm-roberta-large --lang all --profile xlmr --epochs 4 --lr 1.5e-5 --bs 16   # accuracy ceiling check
$P --task topic --name topic_xlmr_all_ca --ckpt xlm-roberta-base --lang all --profile xlmr --epochs 4 --lr 3e-5 --bs 32 --case_aug 0.35   # fixes UPPERCASE invariance
python scripts/s06b_distill.py --teacher topic_xlmr_all_ca --layers 2,5,8,11 --name topic_xlmr_l4_kd   # 4-layer student
python scripts/s06b_distill.py --teacher topic_xlmr_all_ca --layers 3,7,11 --name topic_xlmr_l3_kd     # 3-layer student
python scripts/s02c_gulf_adapt.py --base topic_xlmr_all_ca                                         # Gulf-adapted candidate
echo TRAIN_EXTRA DONE
