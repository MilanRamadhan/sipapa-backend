from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any
import os
import pandas as pd

# ========== PATH SETUP ==========
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ========== LOAD DOC META ==========
DOC_META_DF = pd.read_csv(DATA_DIR / "doc_meta.csv")
DOC_META = {
    int(row["doc_id"]): {
        "doc_id": int(row["doc_id"]),
        "title": row["title"],
        "url": row["url"],
        "doc_len": int(row["doc_len"]),
    }
    for _, row in DOC_META_DF.iterrows()
}

# ========== LOAD CORPUS (VERSI CLEAN v2) ==========
CORPUS_DF = pd.read_csv(DATA_DIR / "corpus_clean_v2.csv")
CORPUS_BY_URL = {
    str(row["url"]): {
        "content_final": str(row.get("content_final", "")),
        "image_url": "" if pd.isna(row.get("image_url")) else str(row.get("image_url", "")),
    }
    for _, row in CORPUS_DF.iterrows()
}

# ========== STOPWORDS + STEMMER (opsional) ==========
STOPWORDS_PATH = BASE_DIR / "stopwords_id.txt"
if STOPWORDS_PATH.exists():
    with STOPWORDS_PATH.open("r", encoding="utf-8") as f:
        STOPWORDS = {line.strip().lower() for line in f if line.strip()}
else:
    STOPWORDS = set()

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    _stemmer = StemmerFactory().create_stemmer()
except Exception:
    _stemmer = None


def _stem(token: str) -> str:
    if _stemmer is None:
        return token
    try:
        return _stemmer.stem(token)
    except Exception:
        return token


# ========== LOAD INVERTED INDEX ==========
with (DATA_DIR / "inverted_index.json").open("r", encoding="utf-8") as f:
    _raw_index = json.load(f)

INVERTED_INDEX: Dict[str, Dict[int, int]] = {}
for term, postings in _raw_index.items():
    doc_tf: Dict[int, int] = {}

    if isinstance(postings, dict):  # direct {doc: tf}
        for doc_id, tf in postings.items():
            doc_tf[int(doc_id)] = int(tf)

    elif isinstance(postings, list):  # list postings
        for item in postings:
            doc_id = None
            tf = 1

            if isinstance(item, (list, tuple)) and len(item) >= 2:
                doc_id, tf = item[0], item[1]

            elif isinstance(item, dict):
                doc_id = item.get("doc_id") or item.get("id") or item.get("doc")
                tf = item.get("tf") or item.get("freq") or item.get("count") or 1

            else:
                doc_id = item
                tf = 1

            if doc_id is not None:
                doc_tf[int(doc_id)] = doc_tf.get(int(doc_id), 0) + int(tf)

    INVERTED_INDEX[term] = doc_tf

DOC_META_MAP = DOC_META
N: int = len(DOC_META_MAP)
AVGDL: float = float(DOC_META_DF["doc_len"].mean())

# ========== IDF CALC ==========

DF_MAP = {term: len(postings) for term, postings in INVERTED_INDEX.items()}

IDF_TFIDF = {}
IDF_BM25 = {}

for term, df in DF_MAP.items():
    df = max(1, df)
    IDF_TFIDF[term] = math.log(N / df)
    IDF_BM25[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)


# ========== QUERY PREPROCESSING ==========

def preprocess_query(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^0-9a-zA-Z\s]", " ", text)
    tokens = re.findall(r"\w+", text)
    return [t for t in tokens if t]


# ========== SEARCH CORE ==========

def _rank_to_results(scores: Counter, top_k: int):
    ranked = scores.most_common(top_k)
    results = []

    for doc_id, score in ranked:
        meta = DOC_META_MAP.get(int(doc_id))
        if not meta:
            continue

        url = meta["url"]
        corpus = CORPUS_BY_URL.get(url, {})
        content_final = corpus.get("content_final", "")
        image_url = corpus.get("image_url", "")

        # Snippet yang pas untuk card area hitam (~850 karakter)
        snippet = content_final[:1005] if content_final else meta["title"]

        results.append({
            "doc_id": doc_id,
            "title": meta["title"],
            "url": url,
            "doc_len": meta["doc_len"],
            "score": float(score),
            "snippet": snippet,
            "image_url": image_url,
        })

    return results


def tfidf_search(query: str, top_k: int = 20):
    tokens = preprocess_query(query)
    scores = Counter()

    for term in tokens:
        postings = INVERTED_INDEX.get(term)
        if not postings:
            continue

        idf = IDF_TFIDF.get(term, 0)
        for doc_id, tf in postings.items():
            scores[doc_id] += tf * idf

    return _rank_to_results(scores, top_k)


def bm25_search(query: str, top_k: int = 20, k1: float = 1.5, b: float = 0.75):
    tokens = preprocess_query(query)
    scores = Counter()

    for term in tokens:
        postings = INVERTED_INDEX.get(term)
        if not postings:
            continue

        idf = IDF_BM25.get(term, 0)

        for doc_id, tf in postings.items():
            dl = DOC_META_MAP[doc_id]["doc_len"]
            denom = tf + k1 * (1 - b + b * (dl / AVGDL))
            score = idf * (tf * (k1 + 1)) / denom
            scores[doc_id] += score

    return _rank_to_results(scores, top_k)


# ========== GET DETAIL DOCUMENT ==========

def get_document(doc_id: int):
    doc_id = int(doc_id)
    meta = DOC_META.get(doc_id)
    if not meta:
        return None

    corpus = CORPUS_BY_URL.get(meta["url"], {})
    content = corpus.get("content_final", "")
    image_url = corpus.get("image_url", "")

    return {
        "doc_id": doc_id,
        "title": meta["title"],
        "url": meta["url"],
        "doc_len": meta["doc_len"],
        "content": content,
        "image_url": image_url,
    }


# ========== METRICS LOADER ==========
def get_metrics():
    path = DATA_DIR / "evaluation_report.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
