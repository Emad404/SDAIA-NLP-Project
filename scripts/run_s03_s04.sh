#!/bin/bash
set -x
python scripts/s03_train_ner.py
python scripts/s04_train_qa.py --ckpt deepset/xlm-roberta-base-squad2 --name qa_xlmr_squad2_zs --zero_shot
python scripts/s04_train_qa.py --ckpt deepset/xlm-roberta-base-squad2 --name qa_xlmr_squad2_ft --epochs 4 --lr 2e-5
python scripts/s04_train_qa.py --ckpt xlm-roberta-base --name qa_xlmr_scratch_ft --epochs 8 --lr 3e-5
echo S03S04 ALL DONE
