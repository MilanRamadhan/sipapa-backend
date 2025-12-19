from flask import Flask, request, jsonify
from flask_cors import CORS

from search_engine import (
    tfidf_search,
    bm25_search,
    get_metrics,
    get_document,
)

from evaluator import evaluate_query_both_algos

app = Flask(__name__)
CORS(app)  # Penting biar frontend (Next.js) bisa akses backend


# ===============================
# Root Test Endpoint
# ===============================
@app.get("/")
def index():
    return jsonify({
        "message": "Sipapa Search Engine API is running!",
        "endpoints": [
            "/search",
            "/metrics",
            "/document/<doc_id>",
            "/evaluate",
        ]
    })


# ===============================
# Search Endpoint
# ===============================
@app.get("/search")
def search():
    # Support both 'query' (local) and 'q' (Vercel) parameters
    query = request.args.get("q", request.args.get("query", "")).strip()
    algo = request.args.get("algo", "tfidf").lower()
    top_k = int(request.args.get("top_k", 20))

    if not query:
        return jsonify({
            "results": [],
            "message": "Query parameter is required",
            "query": "",
            "algo": algo,
            "count": 0
        })

    # Validate algo parameter
    if algo not in ["tfidf", "bm25"]:
        return jsonify({
            "error": "Invalid algorithm. Use 'tfidf' or 'bm25'",
            "requested_algo": algo
        }), 400

    if algo == "bm25":
        results = bm25_search(query, top_k=top_k)
    else:
        results = tfidf_search(query, top_k=top_k)

    return jsonify(results)


# ===============================
# Metrics Endpoint
# ===============================
@app.get("/metrics")
def metrics():
    result = get_metrics()
    return jsonify(result)


# ===============================
# Document Detail Endpoint
# ===============================
@app.get("/document/<int:doc_id>")
def document_detail(doc_id: int):
    """
    Ambil detail satu dokumen berdasarkan doc_id.
    Sekarang kita langsung pakai get_document()
    dan nggak cek len(DOC_META_MAP) lagi,
    karena doc_id di dataset bisa aja nggak 0..N-1.
    """
    doc = get_document(doc_id)
    if not doc:
        return jsonify({
            "error": "Document not found",
            "requested_id": doc_id
        }), 404

    return jsonify(doc)


# ===============================
# Evaluation Endpoint
# ===============================
@app.get("/evaluate")
def evaluate():
    """
    Endpoint untuk evaluasi perbandingan TF-IDF vs BM25
    Params:
        - query: search query string (required)
        - top_k: number of results (default 20)

    Returns:
        {
            "query": "...",
            "tfidf": {...},
            "bm25": {...}
        }
    """
    query = request.args.get("query", "").strip()

    if not query:
        return jsonify({
            "error": "Query parameter is required",
            "usage": "/evaluate?query=<search_term>&top_k=<number>"
        }), 400

    try:
        top_k = int(request.args.get("top_k", 20))
        if top_k < 1 or top_k > 100:
            return jsonify({
                "error": "top_k must be between 1 and 100",
                "requested": top_k
            }), 400
    except ValueError:
        return jsonify({"error": "top_k must be a valid integer"}), 400

    import search_engine as se

    class SearchEngineMock:
        def search(self, q, algo="tfidf", top_k=20):
            if algo == "bm25":
                return se.bm25_search(q, top_k=top_k)
            else:
                return se.tfidf_search(q, top_k=top_k)

    try:
        search_mock = SearchEngineMock()
        result = evaluate_query_both_algos(query, search_mock, top_k=top_k)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "error": "Evaluation failed",
            "message": str(e)
        }), 500


# ===============================
# Evaluation Aggregate Endpoint (Mean dari semua queries)
# ===============================
@app.get("/evaluate/aggregate")
def evaluate_aggregate():
    """
    Evaluasi agregat untuk SEMUA query di ground truth
    Mengembalikan MEAN dari semua metrik (Mean Precision, Mean Recall, Mean F1, MAP)
    """
    from evaluator import evaluate_all_queries
    import search_engine as se

    class SearchEngineMock:
        def search(self, q, algo="tfidf", top_k=20):
            if algo == "bm25":
                return se.bm25_search(q, top_k=top_k)
            else:
                return se.tfidf_search(q, top_k=top_k)

    try:
        # Get top_k from query params (default 20)
        top_k = request.args.get("top_k", default=20, type=int)
        
        search_mock = SearchEngineMock()
        result = evaluate_all_queries(search_mock, top_k=top_k)
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Aggregate evaluation failed",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ===============================
# Run Server
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
