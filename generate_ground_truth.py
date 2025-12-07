"""
Script untuk generate ground truth sederhana berdasarkan keyword matching
Ground truth ini digunakan untuk evaluasi TF-IDF vs BM25
"""
import json
import pandas as pd
import re

# Load corpus
corpus = pd.read_csv("data/corpus_clean.csv")

# Query test dengan keyword yang relevan
test_queries = {
    "pantai bali": ["pantai", "bali"],
    "hotel jakarta": ["hotel", "jakarta"],
    "gunung bromo": ["bromo", "gunung"],
    "wisata bandung": ["wisata", "bandung"],
    "kuliner yogyakarta": ["kuliner", "yogyakarta", "jogja"],
    "danau toba": ["danau", "toba"],
    "candi borobudur": ["candi", "borobudur"],
    "diving bunaken": ["diving", "bunaken", "snorkel"],
    "rafting bali": ["rafting", "bali"],
    "medan": ["medan"],
    "jakarta": ["jakarta"],
    "surabaya": ["surabaya"],
    "bali": ["bali"],
    "jogja": ["jogja", "yogyakarta"],
    "pantai": ["pantai", "beach"],
    "hotel": ["hotel", "penginapan", "resort"],
    "wisata": ["wisata", "destinasi", "tempat"],
    "kuliner": ["kuliner", "makanan", "restoran"],
    "gunung": ["gunung", "pendakian"],
    "museum": ["museum", "sejarah"],
    "taman": ["taman", "park"],
    "pulau": ["pulau", "island"],
    "air terjun": ["air terjun", "waterfall"],
    "danau": ["danau", "lake"],
    "candi": ["candi", "temple"],
}

def generate_relevance_judgments(query_keywords, corpus_df, min_keywords=1):
    """
    Generate relevance judgments berdasarkan keyword matching
    
    Args:
        query_keywords: List of keywords
        corpus_df: DataFrame corpus
        min_keywords: Minimal berapa keyword yang harus muncul
    
    Returns:
        List of relevant doc_ids
    """
    relevant_docs = []
    keywords = [kw.lower() for kw in query_keywords]
    
    for idx, row in corpus_df.iterrows():
        content = str(row.get("content_clean", "")).lower()
        title = str(row.get("title", "")).lower()
        
        # Gabungkan title dan content untuk matching
        text = title + " " + content
        
        # Hitung berapa keyword yang muncul
        keyword_count = sum(1 for kw in keywords if kw in text)
        
        if keyword_count >= min_keywords:
            relevant_docs.append(idx)
    
    return relevant_docs

# Generate ground truth untuk semua query
ground_truth = {}

print("Generating ground truth...")
print("=" * 80)

for query, keywords in test_queries.items():
    relevant_docs = generate_relevance_judgments(keywords, corpus)
    ground_truth[query] = relevant_docs
    print(f"Query: '{query}' -> {len(relevant_docs)} relevant documents")

# Simpan ground truth
with open("data/ground_truth.json", "w", encoding="utf-8") as f:
    json.dump(ground_truth, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 80)
print(f"✅ Ground truth berhasil disimpan ke data/ground_truth.json")
print(f"Total queries: {len(ground_truth)}")
print(f"Total relevant documents: {sum(len(v) for v in ground_truth.values())}")
print(f"Average relevant docs per query: {sum(len(v) for v in ground_truth.values()) / len(ground_truth):.1f}")
