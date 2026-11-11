"""Dataset loaders (all files come from the course 'Data' folder)."""
import json

import pandas as pd

from .config import DATA_DIR


def p(name):
    return DATA_DIR / name


def load_feedback() -> pd.DataFrame:
    return pd.read_csv(p("bayan_feedback.csv"))


def load_ner() -> pd.DataFrame:
    df = pd.read_csv(p("bayan_ner_4000.csv"))
    df["tokens"] = df["tokens_json"].map(json.loads)
    df["tags"] = df["bio_tags_json"].map(json.loads)
    return df


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_qa() -> pd.DataFrame:
    rows = read_jsonl(p("bayan_qa_600.jsonl"))
    df = pd.DataFrame(rows)
    df["answers_text"] = df["answers_text"].map(lambda x: json.loads(x) if isinstance(x, str) else x)
    df["answers_start"] = df["answers_start"].map(lambda x: json.loads(x) if isinstance(x, str) else x)
    return df


def load_qa_smoke() -> pd.DataFrame:
    with open(p("qa_smoke_set.json"), encoding="utf-8") as f:
        df = pd.DataFrame(json.load(f))
    df["answers_text"] = df["answers_text"].map(lambda x: json.loads(x) if isinstance(x, str) else x)
    df["answers_start"] = df["answers_start"].map(lambda x: json.loads(x) if isinstance(x, str) else x)
    return df


def load_queries() -> pd.DataFrame:
    return pd.DataFrame(read_jsonl(p("bayan_queries.jsonl")))


def load_cases() -> pd.DataFrame:
    return pd.read_csv(p("bayan_cases.csv"))


def load_docs() -> pd.DataFrame:
    return pd.read_csv(p("bayan_service_docs.csv"))


def load_pii() -> pd.DataFrame:
    return pd.read_csv(p("pii_test_set.csv"))
