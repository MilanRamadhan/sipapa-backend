"""
Script cepat untuk regenerate indexing dari corpus_clean.csv
Membuat doc_meta.csv dan inverted_index.json
"""
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import math

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

CORPUS_FILE = DATA_DIR / "corpus_clean.csv"
DOC_META_FILE = DATA_DIR / "doc_meta.csv"
INDEX_FILE = DATA_DIR / "inverted_index.json"

print(f"[INFO] Membaca: {CORPUS_FILE}")
df = pd.read_csv(CORPUS_FILE)
print(f"[INFO] Total dokumen: {len(df)}")

# 1. Buat doc_meta.csv
print("\n[1/2] Membuat doc_meta.csv...")
doc_meta = []
for idx, row in df.iterrows():
    doc_meta.append({
        "doc_id": idx,
        "url": row["url"],
        "title": row["title"],
        "image_url": row.get("image_url", ""),
        "doc_len": row["word_count_clean"]
    })

doc_meta_df = pd.DataFrame(doc_meta)
doc_meta_df.to_csv(DOC_META_FILE, index=False)
print(f"     ✓ Saved: {DOC_META_FILE}")

# 2. Buat inverted_index.json (consistent preprocessing)
import re

print("\n[2/2] Membuat inverted_index.json...")
inverted_index = defaultdict(lambda: defaultdict(int))

def preprocess_text(text):
    """Preprocessing konsisten dengan search_engine.py"""
    text = text.lower()
    # Buang URL
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Hanya alphanumeric dan spasi
    text = re.sub(r"[^0-9a-zA-Z\s]", " ", text)
    # Tokenisasi
    tokens = re.findall(r"\w+", text)
    # Filter token pendek
    return [t for t in tokens if len(t) > 1]

for idx, row in df.iterrows():
    content = str(row["content_clean"])
    # Gunakan preprocessing yang sama dengan search
    tokens = preprocess_text(content)
    
    # Count term frequency per document
    for token in tokens:
        inverted_index[token][idx] += 1

# Convert to regular dict untuk JSON
inverted_index_json = {
    term: dict(postings) 
    for term, postings in inverted_index.items()
}

print(f"     → Total unique terms: {len(inverted_index_json)}")

with open(INDEX_FILE, "w", encoding="utf-8") as f:
    json.dump(inverted_index_json, f, ensure_ascii=False)

print(f"     ✓ Saved: {INDEX_FILE}")

print(f"\n[SUCCESS] Indexing selesai!")
print(f"   - Documents: {len(df)}")
print(f"   - Unique terms: {len(inverted_index_json)}")
print(f"   - Avg doc length: {doc_meta_df['doc_len'].mean():.1f} words")
