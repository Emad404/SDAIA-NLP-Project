import json
import os
import random
from pathlib import Path

import pandas as pd
import pytest

from bayan.arabic import model_input, normalize_ar
from bayan.preproc import PREPROC_VERSION, contains_pii, mask_pii, preprocess

HERE = Path(__file__).parent


def test_version_is_semver():
    assert len(PREPROC_VERSION.split(".")) == 3


@pytest.mark.parametrize("raw,expected", json.load(open(HERE / "data" / "golden_preproc.json", encoding="utf-8")))
def test_golden(raw, expected):
    assert preprocess(raw) == expected


def _pii_frames():
    frames = [pd.read_csv(HERE / "data" / "pii_cases.csv")]
    full = os.environ.get("BAYAN_PII_CSV")
    if full and Path(full).exists():
        frames.append(pd.read_csv(full).rename(columns={}))
    return pd.concat(frames, ignore_index=True)


def _digits(s):
    return "".join(ch for ch in str(s) if ch.isdigit())


def test_pii_recall_is_100_percent():
    df = _pii_frames()
    missed = []
    for r in df.itertuples():
        out = preprocess(r.text)
        exp = str(r.expected_value)
        if exp in out or (_digits(exp) and _digits(exp) in _digits(out) and "<" not in out):
            missed.append((r.text, out))
    assert not missed, f"PII leaked: {missed}"


def test_pii_generated_formats():
    rng = random.Random(7)
    bad = []
    for _ in range(300):
        d = "5" + "".join(rng.choice("0123456789") for _ in range(8))
        forms = [f"0{d}", d, f"+966{d}", f"00966{d}", f"0{d[:2]} {d[2:5]} {d[5:]}", f"0{d[:2]}-{d[2:5]}-{d[5:]}",
                 f"+966 {d[:2]} {d[2:5]} {d[5:]}", f"0{d}".translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))]
        for f in forms:
            out = preprocess(f"رقمي {f} شكرا")
            if "<PHONE>" not in out or any(ch.isdigit() for ch in out):
                bad.append((f, out))
        nid = rng.choice("12") + "".join(rng.choice("0123456789") for _ in range(9))
        for f in (nid, nid.translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")), f"{nid[:4]} {nid[4:7]} {nid[7:]}"):
            out = preprocess(f"ID {f} please")
            if "<NATIONAL_ID>" not in out or any(ch.isdigit() for ch in out):
                bad.append((f, out))
        em = f"user{rng.randint(1, 999)}+x@mail{rng.randint(1, 9)}.example.sa"
        if "<EMAIL>" not in preprocess(f"mail {em} now"):
            bad.append((em, "email"))
    assert not bad, bad[:5]


def test_no_false_positive_on_normal_numbers():
    for s in ["Order 12345 was late", "الساعة 5 مساء", "refund within 30 days", "BILL#1234567890123", "15 ديسمبر"]:
        assert not contains_pii(s), s


def test_idempotent():
    for s in ["Call me on 0533113879 sooooo", "الخدمـــة ٣ مرات", "  x  "]:
        assert preprocess(preprocess(s)) == preprocess(s)


def test_raw_is_not_mutated_by_profiles():
    raw = "الخِدْمَةُ إلى مدرسة"
    assert normalize_ar(raw, "xlmr") == raw
    assert normalize_ar(raw, "camelbert") == "الخدمه الي مدرسه"
    assert model_input("أحمد  ", "camelbert") == "احمد"


ARABIC_GOLDEN = [("أحمد", "احمد"), ("مدرسة", "مدرسه"), ("على", "علي"), ("الْكِتَابُ", "الكتاب"),
                 ("إلى", "الي"), ("آمن", "امن"), ("ٱلله", "الله"), ("مُحَمَّد", "محمد"), ("هذه", "هذه"),
                 ("فاتورة", "فاتوره"), ("الخدمة", "الخدمه"), ("وزارة", "وزاره"), ("إدارة", "اداره"),
                 ("أنا", "انا"), ("رخصة", "رخصه"), ("موعد", "موعد"), ("بلدية", "بلديه"), ("مياه", "مياه"),
                 ("كهرباء", "كهرباء"), ("طريق", "طريق"), ("حافلة", "حافله"), ("تأخر", "تاخر"),
                 ("مسؤول", "مسؤول"), ("جدّة", "جده"), ("الرياض", "الرياض"), ("الدمّام", "الدمام"),
                 ("إيصال", "ايصال"), ("اشتراك", "اشتراك"), ("مرحباً", "مرحبا"), ("رمز", "رمز")]


@pytest.mark.parametrize("raw,expected", ARABIC_GOLDEN)
def test_camelbert_profile_golden(raw, expected):
    assert normalize_ar(raw, "camelbert") == expected
