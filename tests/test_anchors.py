from bayan.anchors import CaseAnchors, district_key, month_of, parse, two_stage
import numpy as np


def test_parse_cross_language_equal():
    q = parse("يا جماعة زار المراقب حي النعيم وما وصلني شيء بعدها - حي النعيم، 15 ديسمبر")
    c = parse("The inspector visited Al Naeem and I have heard nothing since. Reported in Al Naeem, Jeddah on 15 December.", "case")
    assert q == c == ("naeem", 15, 12)


def test_typo_and_arabic_digits():
    assert parse("The container smells terrible - Al Safa, 28 Augsut") == ("safa", 28, 8)
    assert parse("x - حي أبحر، ٩ سبتمبر") == ("obhur", 9, 9)


def test_no_anchor():
    assert parse("How do I renew my passport?") == (None, None, None)


def test_two_stage_prefers_anchor_match():
    descs = ["Bus late. Reported in Al Anoud, Dammam on 14 May.", "Bus late. Reported in Al Malqa, Riyadh on 22 January.",
             "الحافلة متأخرة. تم الإبلاغ في حي العنود، الدمام بتاريخ 15 يناير."]
    a = CaseAnchors(descs)
    dense = np.array([0.9, 0.9, 0.85], dtype="float32")
    idx, _ = two_stage(dense, "Bus late - Al Anoud, 15 January", a, {"district": 0.1, "month": 0.1, "day": 0.1}, pool=3)
    assert idx[0] == 2
