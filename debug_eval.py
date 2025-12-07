import requests
import json

query = "wisata"
top_k = 20

print(f"Query: {query}, Top-K: {top_k}\n")

# Get TF-IDF results
tfidf_results = requests.get(f"http://localhost:5000/search?query={query}&algo=tfidf&top_k={top_k}").json()
tfidf_ids = [r['doc_id'] for r in tfidf_results]
print(f"TF-IDF doc_ids: {tfidf_ids}")

# Get BM25 results
bm25_results = requests.get(f"http://localhost:5000/search?query={query}&algo=bm25&top_k={top_k}").json()
bm25_ids = [r['doc_id'] for r in bm25_results]
print(f"BM25 doc_ids:   {bm25_ids}")

# Check overlap
tfidf_set = set(tfidf_ids)
bm25_set = set(bm25_ids)
print(f"\nSame docs: {len(tfidf_set & bm25_set)}/{top_k}")
print(f"TF-IDF only: {tfidf_set - bm25_set}")
print(f"BM25 only: {bm25_set - tfidf_set}")

# Get evaluation
eval_results = requests.get(f"http://localhost:5000/evaluate?query={query}&top_k={top_k}").json()

print(f"\n{'='*60}")
print("EVALUATION RESULTS:")
print(f"{'='*60}")

tfidf_eval = eval_results['tfidf']
bm25_eval = eval_results['bm25']

print(f"\nTF-IDF:")
print(f"  Retrieved: {tfidf_eval['retrieved_count']}")
print(f"  Relevant: {tfidf_eval['relevant_count']}")
print(f"  Relevant Retrieved: {tfidf_eval['relevant_retrieved']}")
print(f"  Precision: {tfidf_eval['precision']:.4f}")
print(f"  Recall: {tfidf_eval['recall']:.4f}")

print(f"\nBM25:")
print(f"  Retrieved: {bm25_eval['retrieved_count']}")
print(f"  Relevant: {bm25_eval['relevant_count']}")
print(f"  Relevant Retrieved: {bm25_eval['relevant_retrieved']}")
print(f"  Precision: {bm25_eval['precision']:.4f}")
print(f"  Recall: {bm25_eval['recall']:.4f}")

# Check if all retrieved docs are in relevant set
if tfidf_eval['relevant_retrieved'] == tfidf_eval['retrieved_count']:
    print(f"\n⚠️  WARNING: ALL TF-IDF retrieved docs are relevant!")
    print(f"   Ground truth terlalu luas: {tfidf_eval['relevant_count']} dokumen")

if bm25_eval['relevant_retrieved'] == bm25_eval['retrieved_count']:
    print(f"\n⚠️  WARNING: ALL BM25 retrieved docs are relevant!")
    print(f"   Ground truth terlalu luas: {bm25_eval['relevant_count']} dokumen")
