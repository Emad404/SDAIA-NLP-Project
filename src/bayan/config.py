"""Paths and global settings. Everything can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_DRIVE_DATA = ("/content/drive/MyDrive/SDAIA NLP/"
               "NATURAL LANGUAGE PROCESSING WITH TRANSFORMER/Data")


def _pick_data_dir() -> Path:
    env = os.environ.get("BAYAN_DATA")
    if env:
        return Path(env)
    for cand in (ROOT / "data", Path(_DRIVE_DATA)):
        if cand.exists():
            return cand
    return ROOT / "data"


DATA_DIR = _pick_data_dir()
WORK_DIR = Path(os.environ.get("BAYAN_WORK", ROOT / "artifacts"))
MODELS_DIR = WORK_DIR / "models"
ART_DIR = WORK_DIR / "artifacts"
REPORTS_DIR = Path(os.environ.get("BAYAN_REPORTS", ROOT / "reports"))
SEED = 42
MAX_LEN_CLS = 128          # serving max length (Day 4: 128 instead of 512)
TRAIN_MAX_LEN = 96

TOPICS = ["account_access", "appointments", "billing", "digital_services",
          "outage", "permits", "transport", "waste"]


def ensure_dirs():
    for p in (WORK_DIR, MODELS_DIR, ART_DIR, REPORTS_DIR):
        p.mkdir(parents=True, exist_ok=True)
