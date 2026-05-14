"""Keyword-based search using BM25 algorithm.

Provides sparse retrieval using the BM25 ranking algorithm for fast
keyword-based document matching.
"""

from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
import pickle
from pathlib import Path

class KeywordSearch:
    """BM25-based keyword search engine.
    
    Implements sparse retrieval using the BM25 algorithm for fast,
    interpretable keyword matching.
    
    Attributes:
        bm25: BM25Okapi ranker instance
        chunks: List of indexed chunks
    """
    def __init__(self):
        """Initialize keyword search engine."""
        self.bm25 = None
        self.chunks = None
        
    
    def build_index(self, chunks: List[Dict]):
        """Build BM25 index from chunks.
        
        Args:
            chunks: List of document chunks to index
        """
        self.chunks = chunks
        
        print(f" Building BM25 keyword index...")
        tokenized_docs = [chunk['content'].lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        print(f" BM25 index built with {len(chunks)} documents")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for top-k chunks using BM25.
        
        Args:
            query: Search query string
            k: Number of results to return
        
        Returns:
            List[Tuple[Dict, float]]: Top k (chunk, score) tuples
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_k_indices = scores.argsort()[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append((self.chunks[idx], float(scores[idx])))
        
        return results
    
    def save(self, path: str):
        """Save BM25 index to disk.
        
        Args:
            path: Directory path to save index
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        with open(save_path / 'bm25.pkl', 'wb') as f:
            pickle.dump({'bm25': self.bm25, 'chunks': self.chunks}, f)
        
        print(f" Saved keyword index to {path}")
        
    
    def load(self, path: str):
        """Load BM25 index from disk.
        
        Args:
            path: Directory path containing saved index
        """
        load_path = Path(path)
        
        with open(load_path / 'bm25.pkl', 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.chunks = data['chunks']
        
        print(f" Loaded keyword index with {len(self.chunks)} documents")