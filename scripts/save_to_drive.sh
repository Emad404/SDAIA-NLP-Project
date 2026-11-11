#!/bin/bash
# Copies the project to Google Drive (Colab-mounted): code + reports + docs + small artefacts always;
# the serving models (topic student + fp32/INT8 ONNX, teacher, sentiment, NER, QA) unless SAVE_MODELS=0.
. /content/env.sh
DEST="/content/drive/MyDrive/SDAIA NLP/Bayan_Project"
W=/content/bayan_work
mkdir -p "$DEST"
rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' /content/bayan_project/ "$DEST/repo/"
mkdir -p "$DEST/work_artifacts"
rsync -a --exclude 'onnx' --exclude 'indexes/*/vectors_all.npy' $W/artifacts/ "$DEST/work_artifacts/artifacts/" 2>/dev/null
rsync -a $W/logs/ "$DEST/work_artifacts/logs/"
if [ "${SAVE_MODELS:-1}" = "1" ]; then
  mkdir -p "$DEST/work_artifacts/artifacts/onnx" "$DEST/work_artifacts/models"
  for m in topic_xlmr_l3_kd; do rsync -a $W/artifacts/onnx/$m/ "$DEST/work_artifacts/artifacts/onnx/$m/"; done
  for m in $(python - <<'PY'
import json
c = json.load(open("/content/bayan_work/artifacts/serving_config.json"))
print(c["topic"]["general"], c["topic"].get("accuracy_reference", ""), c["sentiment"], c["ner"], c["qa"])
PY
  ); do [ -d $W/models/$m ] && rsync -a $W/models/$m/ "$DEST/work_artifacts/models/$m/"; done
fi
echo "saved to: $DEST"; du -sh "$DEST" | cut -f1
