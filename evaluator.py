"""
Modul evaluasi untuk membandingkan algoritma TF-IDF vs BM25
Menggunakan metrik: Precision, Recall, F1, MAP (Mean Average Precision)
"""
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import time
import re


class SearchEvaluator:
    def __init__(self, ground_truth_path="data/ground_truth.json", corpus_path="data/corpus_clean.csv"):
        """
        Initialize evaluator dengan ground truth
        
        Args:
            ground_truth_path: Path ke file ground truth relevance judgments
            corpus_path: Path ke corpus untuk generate ground truth on-the-fly
        """
        try:
            with open(ground_truth_path, "r", encoding="utf-8") as f:
                gt = json.load(f)
                # Convert lists back to sets
                self.ground_truth = {k: set(v) for k, v in gt.items()}
        except FileNotFoundError:
            # Jika ground truth belum ada, gunakan empty dict
            self.ground_truth = {}
        
        # Load corpus untuk generate ground truth on-the-fly
        try:
            self.corpus = pd.read_csv(corpus_path)
            print(f"✓ Corpus loaded: {len(self.corpus)} documents from {corpus_path}")
        except Exception as e:
            print(f"Warning: Failed to load corpus from {corpus_path}: {e}")
            self.corpus = None
    
    def generate_ground_truth_for_query(self, query: str) -> set:
        """
        Generate ground truth on-the-fly untuk query yang tidak ada di ground truth
        Menggunakan TF-IDF scoring untuk menentukan relevansi (top 10% dokumen)
        
        Args:
            query: Query string
        
        Returns:
            Set of relevant doc_ids (top 10% by relevance score)
        """
        if self.corpus is None:
            return set()
        
        # Tokenize query menjadi keywords
        query_lower = query.lower()
        keywords = re.findall(r'\w+', query_lower)
        keywords = [k for k in keywords if len(k) > 2]  # Filter keyword pendek
        
        if not keywords:
            return set()
        
        # Hitung relevance score untuk setiap dokumen
        doc_scores = []
        
        for idx, row in self.corpus.iterrows():
            content = str(row.get("content_clean", "")).lower()
            title = str(row.get("title", "")).lower()
            
            # Scoring dengan bobot berbeda untuk title dan content
            title_score = 0
            content_score = 0
            
            for keyword in keywords:
                # Title match: bobot 5x
                title_matches = title.count(keyword)
                title_score += title_matches * 5
                
                # Content match: bobot 1x
                content_matches = content.count(keyword)
                content_score += content_matches
            
            total_score = title_score + content_score
            
            # Hanya simpan dokumen yang punya score > 0
            if total_score > 0:
                doc_scores.append((idx, total_score))
        
        # Jika tidak ada dokumen dengan score > 0
        if not doc_scores:
            return set()
        
        # Sort by score descending
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Ambil top 5% dokumen sebagai relevant (minimal 20, maksimal 300)
        num_relevant = max(20, min(300, len(doc_scores) // 20))
        relevant_docs = set(doc_id for doc_id, score in doc_scores[:num_relevant])
        
        return relevant_docs
    
    def calculate_precision(self, retrieved: set, relevant: set) -> float:
        """
        Precision = |Retrieved ∩ Relevant| / |Retrieved|
        Proporsi dokumen yang dikembalikan yang relevan
        """
        if len(retrieved) == 0:
            return 0.0
        return len(retrieved & relevant) / len(retrieved)
    
    def calculate_recall(self, retrieved: set, relevant: set) -> float:
        """
        Recall = |Retrieved ∩ Relevant| / |Relevant|
        Proporsi dokumen relevan yang berhasil ditemukan
        """
        if len(relevant) == 0:
            return 0.0
        return len(retrieved & relevant) / len(relevant)
    
    def calculate_f1(self, precision: float, recall: float) -> float:
        """
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        Harmonic mean dari Precision dan Recall
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_average_precision(self, retrieved_ranked: list, relevant: set) -> float:
        """
        Average Precision (AP)
        AP = (1/|Relevant|) * Σ(Precision@k * rel(k))
        
        Args:
            retrieved_ranked: List of doc_ids in ranking order
            relevant: Set of relevant doc_ids
        """
        if len(relevant) == 0:
            return 0.0
        
        relevant_found = 0
        sum_precisions = 0.0
        
        for k, doc_id in enumerate(retrieved_ranked, 1):
            if doc_id in relevant:
                relevant_found += 1
                precision_at_k = relevant_found / k
                sum_precisions += precision_at_k
        
        return sum_precisions / len(relevant)
    
    def calculate_map(self, all_results: dict, ground_truth: dict) -> float:
        """
        Mean Average Precision (MAP)
        MAP = (1/|Q|) * Σ(AP(q))
        
        Args:
            all_results: Dict {query: [doc_id1, doc_id2, ...]}
            ground_truth: Dict {query: set(relevant_doc_ids)}
        """
        ap_scores = []
        
        for query, retrieved in all_results.items():
            if query in ground_truth:
                relevant = ground_truth[query]
                ap = self.calculate_average_precision(retrieved, relevant)
                ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    def evaluate_single_query(self, query: str, retrieved_ids: list) -> dict:
        """
        Evaluasi hasil search untuk satu query
        
        Args:
            query: Query string
            retrieved_ids: List of retrieved doc_ids
        
        Returns:
            Dict dengan metrics: precision, recall, f1, ap
        """
        # Cek apakah ada ground truth untuk query ini
        if query not in self.ground_truth:
            # Generate ground truth on-the-fly jika belum ada
            relevant = self.generate_ground_truth_for_query(query)
            
            # Jika masih tidak ada relevan docs, return 0
            if not relevant:
                return {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "ap": 0.0,
                    "note": "No ground truth available and failed to generate"
                }
            # Simpan ground truth yang baru di-generate
            self.ground_truth[query] = relevant
        
        relevant = self.ground_truth[query]
        retrieved = set(retrieved_ids)
        
        precision = self.calculate_precision(retrieved, relevant)
        recall = self.calculate_recall(retrieved, relevant)
        f1 = self.calculate_f1(precision, recall)
        ap = self.calculate_average_precision(retrieved_ids, relevant)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap": ap,
            "relevant_count": len(relevant),
            "retrieved_count": len(retrieved),
            "relevant_retrieved": len(retrieved & relevant)
        }


def evaluate_query_both_algos(query: str, search_engine, top_k: int = 20) -> dict:
    """
    Evaluasi query menggunakan kedua algoritma (TF-IDF dan BM25)
    Ground truth dibuat secara dinamis dari union hasil kedua algoritma
    
    Args:
        query: Query string
        search_engine: Instance dari SearchEngine
        top_k: Jumlah dokumen yang di-retrieve
    
    Returns:
        Dict dengan hasil evaluasi untuk TF-IDF dan BM25
    """
    evaluator = SearchEvaluator()
    
    # TF-IDF search
    start = time.time()
    tfidf_results = search_engine.search(query, algo="tfidf", top_k=top_k)
    tfidf_time = (time.time() - start) * 1000  # Convert to ms
    tfidf_ids = [r["doc_id"] for r in tfidf_results]
    
    # BM25 search
    start = time.time()
    bm25_results = search_engine.search(query, algo="bm25", top_k=top_k)
    bm25_time = (time.time() - start) * 1000  # Convert to ms
    bm25_ids = [r["doc_id"] for r in bm25_results]
    
    # Hitung overlap antara kedua hasil
    tfidf_set = set(tfidf_ids)
    bm25_set = set(bm25_ids)
    overlap = len(tfidf_set & bm25_set)
    overlap_percentage = (overlap / top_k) * 100 if top_k > 0 else 0
    
    # Buat ground truth dinamis - gunakan scoring yang lebih ketat
    # Hanya dokumen dengan score tinggi yang dianggap relevan
    query_lower = query.lower()
    keywords = re.findall(r'\w+', query_lower)
    keywords = [k for k in keywords if len(k) > 2]
    
    relevant_docs = set()
    doc_scores = {}
    
    # Score semua dokumen dari hasil TF-IDF
    for i, doc_id in enumerate(tfidf_ids):
        try:
            if evaluator.corpus is not None:
                row = evaluator.corpus.loc[doc_id]
                content = str(row.get("content_clean", "")).lower()
                title = str(row.get("title", "")).lower()
                
                score = 0
                for keyword in keywords:
                    score += title.count(keyword) * 10  # Title sangat penting
                    score += content.count(keyword) * 2
                
                # Boost untuk ranking tinggi di TF-IDF
                position_boost = (top_k - i) / top_k * 5
                score += position_boost
                doc_scores[doc_id] = score
        except:
            continue
    
    # Score semua dokumen dari hasil BM25
    for i, doc_id in enumerate(bm25_ids):
        try:
            if evaluator.corpus is not None:
                row = evaluator.corpus.loc[doc_id]
                content = str(row.get("content_clean", "")).lower()
                title = str(row.get("title", "")).lower()
                
                score = 0
                for keyword in keywords:
                    score += title.count(keyword) * 10
                    score += content.count(keyword) * 2
                
                # Boost untuk ranking tinggi di BM25
                position_boost = (top_k - i) / top_k * 5
                
                if doc_id in doc_scores:
                    # Average jika dokumen muncul di kedua hasil
                    doc_scores[doc_id] = (doc_scores[doc_id] + score + position_boost) / 2
                else:
                    doc_scores[doc_id] = score + position_boost
        except:
            continue
    
    # Ambil dokumen dengan score di atas median sebagai relevan
    if doc_scores:
        scores_list = list(doc_scores.values())
        threshold = np.median(scores_list) if len(scores_list) > 0 else 0
        relevant_docs = set(doc_id for doc_id, score in doc_scores.items() if score >= threshold)
    
    # Fallback: jika tidak ada yang relevan, gunakan top 60% dokumen
    if not relevant_docs:
        all_docs = list(set(tfidf_ids + bm25_ids))
        cutoff = int(len(all_docs) * 0.6)
        relevant_docs = set(all_docs[:cutoff])
    
    # Override ground truth
    evaluator.ground_truth[query] = relevant_docs
    
    # Evaluasi TF-IDF
    tfidf_eval = evaluator.evaluate_single_query(query, tfidf_ids)
    tfidf_eval["runtime"] = tfidf_time
    
    # Evaluasi BM25
    bm25_eval = evaluator.evaluate_single_query(query, bm25_ids)
    bm25_eval["runtime"] = bm25_time
    
    # Hitung Average Precision untuk ranking quality
    tfidf_eval["ap"] = evaluator.calculate_average_precision(tfidf_ids, relevant_docs)
    bm25_eval["ap"] = evaluator.calculate_average_precision(bm25_ids, relevant_docs)
    
    return {
        "query": query,
        "tfidf": tfidf_eval,
        "bm25": bm25_eval,
        "overlap_docs": overlap,
        "overlap_percentage": round(overlap_percentage, 2),
        "total_relevant": len(relevant_docs)
    }
