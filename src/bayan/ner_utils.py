"""NER helpers shared by training (s03), evaluation and the service: word tokenisation, BIO decoding."""
import re

from .preproc import preprocess

# words keep <PHONE>-style placeholders, hyphens/apostrophes; every other symbol is its own token
_WORD = re.compile(r"<[A-Z_]+>|[\w'’\-/@.]+(?<![.])|[^\w\s]", re.UNICODE)
CLITICS_AR = ["و", "ب", "ل", "ف", "وب", "ول", "فب"]


def words_with_spans(text: str):
    return [(m.group(), m.start(), m.end()) for m in _WORD.finditer(text)]


def prep_token(tok: str) -> str:
    out = preprocess(tok)
    return out if out.strip() else tok


def bio_to_spans(words, tags):
    """words: list[(w, s, e)], tags: list[str] -> list[{type,text,start,end}] (character offsets in the cleaned text)."""
    spans, cur = [], None
    for (w, s, e), t in zip(words, tags):
        if t == "O":
            if cur:
                spans.append(cur)
                cur = None
            continue
        pre, typ = t.split("-", 1)
        if pre == "B" or cur is None or cur["type"] != typ:
            if cur:
                spans.append(cur)
            cur = {"type": typ, "start": s, "end": e}
        else:
            cur["end"] = e
    if cur:
        spans.append(cur)
    return spans
