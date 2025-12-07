from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import json
import csv
import math
from collections import defaultdict
import traceback

# =====================================
# GLOBAL STATE
# =====================================
INVERTED_INDEX = {}
DOC_META = {}
DOC_LENGTHS = {}
DOC_CONTENT = {}  # <-- isi dari corpus_clean_v2.csv (content_final)
TOTAL_DOCS = 0
INIT_ERROR = None


def log(msg: str):
    print("[search.py]", msg)


# =====================================
# LOAD INVERTED INDEX
# format: { "kompas": { "0": 1, "1": 3, ... }, ... }
# =====================================
try:
    log("Loading data/inverted_index.json ...")
    with open("data/inverted_index.json", "r", encoding="utf-8") as f:
        INVERTED_INDEX = json.load(f)
    log(f"inverted_index has {len(INVERTED_INDEX)} terms")
except Exception as e:
    INIT_ERROR = f"Failed to load inverted_index.json: {e}"
    traceback.print_exc()


# =====================================
# LOAD DOC META
# doc_meta.csv:
# doc_id,url,title,image_url,doc_len
# =====================================
try:
    log("Loading data/doc_meta.csv ...")
    with open("data/doc_meta.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = str(row.get("doc_id"))
            if not doc_id or doc_id == "None":
                continue

            DOC_META[doc_id] = {
                "url": row.get("url", ""),
                "title": row.get("title", "Untitled"),
                "image_url": row.get("image_url", ""),
                "doc_len": int(row.get("doc_len", 1) or 1),
            }
            DOC_LENGTHS[doc_id] = DOC_META[doc_id]["doc_len"]

    TOTAL_DOCS = len(DOC_META)

    # fallback kalau meta kosong tapi index ada
    if TOTAL_DOCS == 0 and INVERTED_INDEX:
        all_ids = set()
        for postings in INVERTED_INDEX.values():
            for d in postings.keys():
                all_ids.add(str(d))
        TOTAL_DOCS = len(all_ids)

    log(f"doc_meta has {TOTAL_DOCS} docs")
except FileNotFoundError:
    log("doc_meta.csv not found, DOC_META is empty")
except Exception as e:
    INIT_ERROR = f"Failed to load doc_meta.csv: {e}"
    traceback.print_exc()


# =====================================
# LOAD CORPUS CONTENT
# corpus_clean_v2.csv:
# url,title,image_url,content_final
# diasumsikan urutannya sama dgn doc_meta → index baris = doc_id
# =====================================
try:
    log("Loading data/corpus_clean_v2.csv ...")
    with open("data/corpus_clean_v2.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            doc_id = str(idx)
            DOC_CONTENT[doc_id] = (row.get("content_final") or "").strip()
    log(f"corpus_clean_v2 has {len(DOC_CONTENT)} docs")
except FileNotFoundError:
    log("corpus_clean_v2.csv not found, DOC_CONTENT is empty (detail view will have no content)")
except Exception as e:
    # jangan matikan semuanya, cukup log aja
    traceback.print_exc()
    log(f"Failed to load corpus_clean_v2.csv: {e}")


# =====================================
# BM25 SEARCH
# =====================================
def bm25_search(query_terms, top_k=20):
    """
    query_terms: list token lower-case
    """
    if not INVERTED_INDEX:
        return []

    # parameter BM25
    k1 = 1.5
    b = 0.75

    N = TOTAL_DOCS or 1

    # average doc length
    if DOC_LENGTHS:
        avg_dl = sum(DOC_LENGTHS.values()) / len(DOC_LENGTHS)
    else:
        avg_dl = 300.0  # fallback asal

    scores = defaultdict(float)

    for term in query_terms:
        postings = INVERTED_INDEX.get(term)
        if not postings:
            continue

        # postings: dict {doc_id: freq}
        df = len(postings)
        if df == 0:
            continue

        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

        for raw_doc_id, freq in postings.items():
            doc_id = str(raw_doc_id)
            tf = int(freq)

            dl = DOC_LENGTHS.get(doc_id, avg_dl)

            score = idf * ((tf * (k1 + 1)) /
                           (tf + k1 * (1 - b + b * (dl / avg_dl))))
            scores[doc_id] += score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for doc_id, score in ranked[:top_k]:
        meta = DOC_META.get(doc_id, {})
        results.append({
            "doc_id": doc_id,
            "title": meta.get("title", "Untitled"),
            "url": meta.get("url", ""),
            "image_url": meta.get("image_url", ""),
            "doc_len": meta.get("doc_len", None),
            "score": score,
        })

    return results


# =====================================
# HTTP HANDLER UNTUK VERCEL
# =====================================
class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            if INIT_ERROR is not None:
                body = json.dumps({
                    "error": "INIT_ERROR",
                    "message": INIT_ERROR,
                }).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)
                return

            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)

            # =========================================
            # 1) DETAIL MODE: ?doc_id=123
            # =========================================
            doc_id_param = params.get("doc_id", [None])[0]
            if doc_id_param is not None:
                try:
                    doc_id = str(int(doc_id_param))
                except ValueError:
                    doc_id = str(doc_id_param)

                meta = DOC_META.get(doc_id)
                content = DOC_CONTENT.get(doc_id, "")

                if not meta and not content:
                    body = json.dumps({
                        "error": "DOCUMENT_NOT_FOUND",
                        "requested_id": doc_id,
                    }).encode("utf-8")
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(body)
                    return

                doc = {
                    "doc_id": doc_id,
                    "title": (meta or {}).get("title", "Untitled"),
                    "url": (meta or {}).get("url", ""),
                    "image_url": (meta or {}).get("image_url", ""),
                    "doc_len": (meta or {}).get("doc_len"),
                    "content": content,
                }

                body = json.dumps(doc, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)
                return

            # =========================================
            # 2) SEARCH MODE: ?q=...&top_k=...
            # =========================================
            q = params.get("q", [""])[0].strip()
            top_k_str = params.get("top_k", ["20"])[0]

            if not q:
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    b'{"error":"Query parameter q is required","results":[]}'
                )
                return

            try:
                top_k = int(top_k_str)
            except ValueError:
                top_k = 20

            query_terms = q.lower().split()
            results = bm25_search(query_terms, top_k=top_k)

            body = json.dumps({
                "query": q,
                "count": len(results),
                "results": results,
            }, ensure_ascii=False).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)

        except Exception as e:
            traceback.print_exc()
            body = json.dumps({
                "error": "RUNTIME_ERROR",
                "message": str(e),
            }).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
