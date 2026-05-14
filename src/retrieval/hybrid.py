"""Hybrid search combining vector and keyword-based retrieval.

Combines dense vector search with sparse BM25 keyword search for improved
retrieval quality and recall.
"""

from typing import List, Dict, Tuple
from .vector_search import VectorSearch
from .keyword_search import KeywordSearch
from .reranker import CrossEncoderReranker

class HybridSearch:
    """Combine vector and keyword search for retrieval.
    
    Merges results from both search methods with configurable weights
    and optional cross-encoder reranking.
    
    Attributes:
        vector_search: Vector similarity search engine
        keyword_search: BM25 keyword search engine
        vector_weight: Weight for vector search results (default: 0.7)
        keyword_weight: Weight for keyword search results (default: 0.3)
        reranker: Optional cross-encoder for reranking results
    """
    def __init__(self, vector_weight: float = 0.7, keyword_weight: float = 0.3, use_reranker: bool = False):
        """Initialize hybrid search engine.
        
        Args:
            vector_weight: Weight for vector search scores (default: 0.7)
            keyword_weight: Weight for keyword search scores (default: 0.3)
            use_reranker: Whether to use cross-encoder reranking (default: False)
        """
        self.vector_search = VectorSearch()
        self.keyword_search = KeywordSearch()
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.reranker = CrossEncoderReranker() if use_reranker else None
    
    def build_index(self, chunks: List[Dict]):
        """Build both vector and keyword indices.
        
        Args:
            chunks: List of document chunks to index
        """
        print("\n🔧 Building hybrid search index...")
        print("="*60)
        self.vector_search.build_index(chunks)
        self.keyword_search.build_index(chunks)
        print("="*60)
        print("Hybrid index complete!")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """Search using hybrid vector + keyword approach.
        
        Args:
            query: Search query string
            k: Number of results to return (default: 5)
        
        Returns:
            List[Tuple[Dict, float]]: Top k (chunk, score) tuples
        """
        vector_results = self.vector_search.search(query, k=k*2)
        keyword_results = self.keyword_search.search(query, k=k*2)
        
        vector_scores = self._normalize([s for _, s in vector_results])
        keyword_scores = self._normalize([s for _, s in keyword_results])
        
        combined = {}
        
        for (chunk, _), norm_score in zip(vector_results, vector_scores):
            cid = chunk['chunk_id']
            combined[cid] = {
                'chunk': chunk,
                'score': self.vector_weight * norm_score
            }
        
        for (chunk, _), norm_score in zip(keyword_results, keyword_scores):
            cid = chunk['chunk_id']
            if cid in combined:
                combined[cid]['score'] += self.keyword_weight * norm_score
            else:
                combined[cid] = {
                    'chunk': chunk,
                    'score': self.keyword_weight * norm_score
                }
        
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:k*2]
        
        results = [(item['chunk'], item['score']) for item in sorted_results]
        
        # Apply cross-encoder reranking if enabled
        if self.reranker:
            results = self.reranker.rerank(query, results, top_k=k)
        else:
            results = results[:k]
        
        return results
    
    def _normalize(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range.
        
        Args:
            scores: List of scores to normalize
        
        Returns:
            List[float]: Normalized scores (0-1)
        """
        if not scores:
            return []
        min_s = min(scores)
        max_s = max(scores)
        if max_s == min_s:
            return [1.0] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]
    
    def save(self, path: str):
        """Save both indices to disk.
        
        Args:
            path: Directory path to save indices
        """
        self.vector_search.save(path)
        self.keyword_search.save(path)
        print(f" Saved complete hybrid index to {path}")
    
    def load(self, path: str):
        """Load both indices from disk.
        
        Args:
            path: Directory path containing saved indices
        """
        self.vector_search.load(path)
        self.keyword_search.load(path)
        print(f" Loaded complete hybrid index from {path}")