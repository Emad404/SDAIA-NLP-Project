"""Stage-2 'anchor' re-scorer for similar-case retrieval.

Every case description ends with a district and a date ("... Reported in Al Anoud, Dammam on 15 January." /
"... تم الإبلاغ في حي العنود، الدمام بتاريخ 15 يناير."). Citizens usually repeat them in their query
("... - Al Anoud, 15 January"). A bi-encoder is nearly blind to these tokens, so near-duplicate cases that differ only in
district/date tie. Stage 2 parses (district, day, month) from query and candidate (language-independent canonical keys)
and adds a bonus to the dense cosine:  score = cos + w_district*[district==] + w_month*[month==] + w_day*[day==].
The weights are tuned on DEV queries only (scripts/s05_search.py) and stored in search_config.json.
"""
import difflib
import re

import numpy as np

_DIG = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
MONTHS = {
    1: ["january", "jan", "يناير"], 2: ["february", "feb", "فبراير"], 3: ["march", "mar", "مارس"], 4: ["april", "apr", "أبريل", "ابريل"],
    5: ["may", "مايو"], 6: ["june", "jun", "يونيو"], 7: ["july", "jul", "يوليو"], 8: ["august", "aug", "أغسطس", "اغسطس"],
    9: ["september", "sep", "sept", "سبتمبر"], 10: ["october", "oct", "أكتوبر", "اكتوبر"], 11: ["november", "nov", "نوفمبر"],
    12: ["december", "dec", "ديسمبر"],
}
_MON = {w: m for m, ws in MONTHS.items() for w in ws}
# canonical id -> [english, arabic]. Coverage against the real case base is verified by scripts/s05_search.py.
DISTRICTS = {   # the 17 districts present in bayan_cases.csv (verified by listing them; coverage is re-checked in s05)
    "olaya": ["Al Olaya", "العليا"], "malqa": ["Al Malqa", "الملقا"], "hittin": ["Hittin", "حطين"], "nakheel": ["Al Nakheel", "النخيل"],
    "naeem": ["Al Naeem", "النعيم"], "hamra": ["Al Hamra", "الحمراء"], "safa": ["Al Safa", "الصفا"], "salamah": ["Al Salamah", "السلامة"],
    "obhur": ["Obhur", "أبحر"], "anoud": ["Al Anoud", "العنود"], "badiyah": ["Al Badiyah", "البادية"],
    "sulaimaniyah": ["As Sulaimaniyah", "السليمانية"], "narjis": ["An Narjis", "النرجس"], "rawdah": ["Al Rawdah", "الروضة"],
    "yasmin": ["Al Yasmin", "الياسمين"], "faisaliyah": ["Al Faisaliyah", "الفيصلية"], "shati": ["Al Shati", "الشاطئ"],
    "uhud": ["Uhud", "أحد"],
}


def _norm_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"^(حي|district|al|an|as|ash|at|ad|ar|az|el)\s+", "", s)
    s = re.sub(r"^ال", "", s)
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه").replace("ى", "ي")
    return re.sub(r"[^\w]", "", s)


_LEX = {}
for _k, (_en, _ar) in DISTRICTS.items():
    _LEX[_norm_name(_en)] = _k
    _LEX[_norm_name(_ar)] = _k


def district_key(name: str):
    n = _norm_name(name)
    if not n:
        return None
    if n in _LEX:
        return _LEX[n]
    close = difflib.get_close_matches(n, list(_LEX), n=1, cutoff=0.85)   # tolerate typos
    return _LEX[close[0]] if close else "?" + n          # unknown district: still comparable within one script


def month_of(tok: str):
    t = tok.strip().lower().strip(".,،")
    if t in _MON:
        return _MON[t]
    close = difflib.get_close_matches(t, list(_MON), n=1, cutoff=0.8)
    return _MON[close[0]] if close else None


_CASE_RE = re.compile(r"(?:Reported in|الإبلاغ في)\s+(?:حي\s+)?(?P<d>[^,،]+?)\s*[,،]\s*[^,،]+?\s+(?:on|بتاريخ)\s+(?P<day>\d{1,2})\s+(?P<mon>[^\s.،,]+)", re.I)
_QRY_RE = re.compile(r"[-–—]\s*(?:حي\s+)?(?P<d>[^,،\-–—]+?)\s*[,،]\s*(?P<day>\d{1,2})\s+(?P<mon>[^\s.،,]+)")


def parse(text: str, kind: str = "query"):
    """-> (district_key|None, day|None, month|None)"""
    t = (text or "").translate(_DIG)
    m = (_CASE_RE if kind == "case" else _QRY_RE).search(t)
    if not m:
        m = (_QRY_RE if kind == "case" else _CASE_RE).search(t)
    if not m:
        return (None, None, None)
    return (district_key(m.group("d")), int(m.group("day")), month_of(m.group("mon")))


class CaseAnchors:
    """Vectorised anchors for the whole case base (built once at startup from the raw descriptions)."""

    def __init__(self, descriptions):
        P = [parse(d, "case") for d in descriptions]
        self.dist = np.array([p[0] or "" for p in P], dtype=object)
        self.day = np.array([p[1] or 0 for p in P], dtype=np.int16)
        self.mon = np.array([p[2] or 0 for p in P], dtype=np.int16)
        self.coverage = float(np.mean([p[0] is not None for p in P]))
        self.unknown_districts = sorted({p[0] for p in P if p[0] and p[0].startswith("?")})

    def exact_matches(self, qa):
        d, day, mon = qa
        if d is None or day is None or mon is None:
            return np.array([], dtype=int)
        return np.where((self.dist == d) & (self.day == day) & (self.mon == mon))[0]

    def bonus(self, idx, qa, w):
        d, day, mon = qa
        idx = np.asarray(idx)
        b = np.zeros(len(idx), dtype="float32")
        if d is not None:
            b += w["district"] * (self.dist[idx] == d)
        if mon is not None:
            b += w["month"] * (self.mon[idx] == mon)
        if day is not None:
            b += w["day"] * (self.day[idx] == day)
        return b


def two_stage(dense_scores_all, qtext_raw, anchors: CaseAnchors, w: dict, pool=300, topn=100):
    """dense_scores_all: cosine to every case (len N). Returns (idx_ranked, scores_ranked). Stage 1 = dense top-`pool`
    plus every case whose anchor equals the query's; stage 2 = anchor-bonus re-scoring."""
    cand = np.argpartition(-dense_scores_all, min(pool, len(dense_scores_all) - 1))[:pool]
    qa = parse(qtext_raw, "query")
    extra = anchors.exact_matches(qa)
    if len(extra):
        # only add the exact-anchor cases that are at least moderately similar (keeps the pool small and on-topic)
        extra = extra[np.argsort(-dense_scores_all[extra])[:pool]]
        cand = np.union1d(cand, extra)
    sc = dense_scores_all[cand] + anchors.bonus(cand, qa, w)
    order = np.argsort(-sc)[:topn]
    return cand[order], sc[order]
