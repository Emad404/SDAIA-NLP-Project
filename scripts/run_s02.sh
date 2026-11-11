#!/bin/bash
# Runs all classifier trainings sequentially (logs to $BAYAN_WORK/logs/s02.log)
set -x
P="python scripts/s02_train_cls.py"
$P --task topic --name topic_xlmr_all --ckpt xlm-roberta-base --lang all --profile xlmr
$P --task topic --name topic_camelbert_mix_ar --ckpt CAMeL-Lab/bert-base-arabic-camelbert-mix --lang ar --profile camelbert
$P --task topic --name topic_camelbert_da_ar --ckpt CAMeL-Lab/bert-base-arabic-camelbert-da --lang ar --profile camelbert
$P --task topic --name topic_xlmr_ar --ckpt xlm-roberta-base --lang ar --profile xlmr
$P --task topic --name topic_xlmr_en --ckpt xlm-roberta-base --lang en --profile xlmr
$P --task sentiment --name sent_xlmr_all --ckpt xlm-roberta-base --lang all --profile xlmr
$P --task sentiment --name sent_camelbert_da_ar --ckpt CAMeL-Lab/bert-base-arabic-camelbert-da --lang ar --profile camelbert
echo S02 ALL DONE
