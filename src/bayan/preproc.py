"""One versioned preprocessing module, used by training, evaluation AND serving.

Order (Day 1 recipe): strip invisible chars -> MASK PII -> NFC -> drop tatweel ->
collapse 3+ repeated letters -> collapse whitespace. Emoji, punctuation and case are kept.
Arabic-Indic digits are recognised inside PII (phones / national IDs) but are left
untouched elsewhere.
"""
import re
import unicodedata

PREPROC_VERSION = "1.0.0"

_ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩" + "۰۱۲۳۴۵۶۷۸۹"
_DIGIT_MAP = str.maketrans(_ARABIC_INDIC, "0123456789" * 2)  # 1-to-1 => offsets preserved

_INVISIBLE = re.compile("[​-‏‪-‮⁦-⁩﻿]")
_REPEAT = re.compile(r"([^\W\d_])\1{2,}")
_WS = re.compile(r"\s+")

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# Saudi mobile: 05xxxxxxxx | 5xxxxxxxx | +9665xxxxxxxx | 009665xxxxxxxx | +966 (0) 5..., separators allowed
_PHONE = re.compile(r"(?<!\d)(?:(?:\+|00)966[\s\-]?\(?0?\)?[\s\-]?|0)?5(?:[\s\-]?\d){8}(?!\d)")
# National ID / iqama: 10 digits starting with 1 or 2, separators allowed
_NID = re.compile(r"(?<!\d)[12](?:[\s\-]?\d){9}(?!\d)")

_PATTERNS = [("EMAIL", _EMAIL), ("PHONE", _PHONE), ("NATIONAL_ID", _NID)]
PLACEHOLDERS = ("<EMAIL>", "<PHONE>", "<NATIONAL_ID>")


def _find_spans(text: str):
    mapped = text.translate(_DIGIT_MAP)
    cands = []
    for pri, (label, rx) in enumerate(_PATTERNS):
        for mo in rx.finditer(mapped):
            cands.append((mo.start(), mo.end(), pri, label))
    cands.sort(key=lambda c: (c[0], c[2], -(c[1] - c[0])))
    spans, last_end = [], -1
    for s, e, _, label in cands:
        if s >= last_end:
            spans.append((s, e, label))
            last_end = e
    return spans


def mask_pii(text) -> str:
    text = "" if text is None else str(text)
    spans = _find_spans(text)
    if not spans:
        return text
    out, pos = [], 0
    for s, e, label in spans:
        out.append(text[pos:s])
        out.append(f"<{label}>")
        pos = e
    out.append(text[pos:])
    return "".join(out)


def normalize(text) -> str:
    t = unicodedata.normalize("NFC", "" if text is None else str(text))
    t = t.replace("ـ", "")                 # tatweel
    t = _INVISIBLE.sub("", t)
    t = _REPEAT.sub(r"\1\1", t)                 # sooooo -> soo, جداااا -> جداا
    return _WS.sub(" ", t).strip()


def preprocess(text) -> str:
    """Mask PII first, then normalise. This is THE preprocessing function."""
    t = "" if text is None else str(text)
    if t != t:  # NaN
        return ""
    t = _INVISIBLE.sub("", t)
    return normalize(mask_pii(t))


def contains_pii(text) -> bool:
    return bool(_find_spans("" if text is None else str(text)))
