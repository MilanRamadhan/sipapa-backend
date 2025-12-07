import pandas as pd
import re

# Load corpus
corpus = pd.read_csv("data/corpus_clean.csv")
print(f"Corpus loaded: {len(corpus)} documents")
print(f"Columns: {corpus.columns.tolist()}")
print(f"\nFirst few rows:")
print(corpus.head())

# Test generate ground truth
query = "medan"
query_lower = query.lower()
keywords = re.findall(r'\w+', query_lower)
keywords = [k for k in keywords if len(k) > 2]

print(f"\nQuery: {query}")
print(f"Keywords: {keywords}")

# Hitung relevance score
doc_scores = []

for idx, row in corpus.iterrows():
    content = str(row.get("content_clean", "")).lower()
    title = str(row.get("title", "")).lower()
    
    title_score = 0
    content_score = 0
    
    for keyword in keywords:
        title_matches = title.count(keyword)
        title_score += title_matches * 5
        
        content_matches = content.count(keyword)
        content_score += content_matches
    
    total_score = title_score + content_score
    
    if total_score > 0:
        doc_scores.append((idx, total_score, title[:50]))

print(f"\nTotal docs with score > 0: {len(doc_scores)}")

# Show top 10
doc_scores.sort(key=lambda x: x[1], reverse=True)
print("\nTop 10 documents by score:")
for doc_id, score, title_snippet in doc_scores[:10]:
    print(f"  doc_id={doc_id}, score={score}, title={title_snippet}")

# Calculate how many would be relevant
num_relevant = max(20, min(300, len(doc_scores) // 20))
print(f"\nRelevant docs threshold (5%): {num_relevant}")
