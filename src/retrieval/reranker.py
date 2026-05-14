"""Cross-encoder based reranking for improved retrieval.

Uses cross-encoder models to rerank retrieval results based on query-document
relevance for higher quality results.
"""

from sentence_transformers import CrossEncoder
from typing import List, Dict, Tuple


class CrossEncoderReranker:
    """Rerank retrieval results using cross-encoder model.
    
    Uses a cross-encoder model to score query-document pairs and rerank
    results for improved relevance.
    
    Attributes:
        model: CrossEncoder model for relevance scoring
    """
    def __init__(self, model_name: str = 'BAAI/bge-reranker-base'):
        """Initialize cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model name (default: BAAI/bge-reranker-base)
        """
        print(f" Loading cross-encoder model: {model_name}...")
        self.model = CrossEncoder(model_name)
        print(f" Cross-encoder model loaded")
    
    def rerank(
        self, 
        query: str, 
        results: List[Tuple[Dict, float]], 
        top_k: int = None
    ) -> List[Tuple[Dict, float]]:
        """Rerank retrieval results using cross-encoder model.
        
        Args:
            query: Original search query
            results: List of (chunk, score) tuples to rerank
            top_k: Number of top results to return (None for all)
        
        Returns:
            List[Tuple[Dict, float]]: Reranked (chunk, score) tuples
        """
        
        if not results:
            return results
        
        if top_k is None:
            top_k = len(results)
        
        # Extract chunks and compute relevance scores
        chunks = [chunk for chunk, _ in results]
        texts = [chunk['content'] for chunk in chunks]
        
        print(f" Reranking {len(results)} results...")
        
        # Create query-document pairs for cross-encoder
        pairs = [[query, text] for text in texts]
        
        # Get reranked scores
        scores = self.model.predict(pairs)
        
        # Combine chunks with reranked scores and sort
        reranked_results = [
            (chunk, float(score)) 
            for chunk, score in zip(chunks, scores)
        ]
        
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        print(f" Reranking complete")
        
        return reranked_results[:top_k]
