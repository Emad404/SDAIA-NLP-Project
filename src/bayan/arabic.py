"""Per-checkpoint Arabic normalisation profiles (Day 3). The raw text is never modified;
only the model-input copy is. Profile names are stored next to every model."""
import re

from .preproc import preprocess

_ALEF = {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"}
_CAMEL_MAP = str.maketrans({**_ALEF, "ى": "ي", "ة": "ه"})
_DIACRITICS = re.compile("[ً-ٰٟ]")

PROFILES = ("xlmr", "camelbert")


def normalize_ar(text: str, profile: str) -> str:
    """Apply the Arabic-specific steps a checkpoint expects (after shared preprocessing)."""
    if profile == "camelbert":
        text = text.translate(_CAMEL_MAP)
        return _DIACRITICS.sub("", text)
    if profile in ("xlmr", "none", None):
        return text
    raise ValueError(f"unknown Arabic profile: {profile}")


def model_input(text, profile: str = "xlmr") -> str:
    """shared preprocessing (mask PII, NFC, ...) then the checkpoint's own Arabic profile."""
    return normalize_ar(preprocess(text), profile)


def arabic_ratio(text: str) -> float:
    letters = [c for c in str(text) if c.isalpha()]
    if not letters:
        return 0.0
    return sum("؀" <= c <= "ۿ" for c in letters) / len(letters)


def detect_language(text: str) -> str:
    return "ar" if arabic_ratio(text) >= 0.4 else "en"
