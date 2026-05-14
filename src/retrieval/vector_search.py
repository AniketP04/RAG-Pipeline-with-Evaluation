"""Vector-based semantic search using embeddings.

Provides dense retrieval using sentence embeddings and FAISS vector index
for semantic similarity matching.
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Tuple
import pickle
from pathlib import Path


class VectorSearch:
    """Semantic search using vector embeddings and FAISS index.
    
    Uses sentence transformers for embedding text and FAISS for
    efficient similarity search.
    
    Attributes:
        model: SentenceTransformer model for embeddings
        dimension: Embedding dimension
        index: FAISS vector index
        chunks: List of indexed chunks
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize vector search with embedding model.
        
        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2)
        """
        print(f" Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = None
        print(f" Model loaded (dimension: {self.dimension})")
    
    
    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Generate embeddings for chunks.
        
        Args:
            chunks: List of chunks to embed
        
        Returns:
            np.ndarray: Float32 embedding matrix (n_chunks, embedding_dim)
        """
        texts = [chunk['content'] for chunk in chunks]
        
        print(f" Generating embeddings for {len(texts)} chunks...")
        print("This takes ~10-15 minutes on CPU...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        return embeddings.astype('float32')
    
    def build_index(self, chunks: List[Dict]):
        """Build FAISS index from chunks.
        
        Args:
            chunks: List of document chunks to index
        """
        self.chunks = chunks
        embeddings = self.create_embeddings(chunks)
        
        print(f"🏗️ Building FAISS index...")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        
        print(f" Index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for top-k similar chunks.
        
        Args:
            query: Search query string
            k: Number of results to return
        
        Returns:
            List[Tuple[Dict, float]]: Top k (chunk, similarity_score) tuples
        """
        query_embedding = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity = 1 / (1 + dist)
            results.append((self.chunks[idx], similarity))
        
        return results
    
    def save(self, path: str):
        """Save FAISS index and chunks to disk.
        
        Args:
            path: Directory path to save index
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(save_path / 'faiss.index'))
        
        with open(save_path / 'chunks.pkl', 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print(f" Saved vector index to {path}")
    
    def load(self, path: str):
        """Load FAISS index and chunks from disk.
        
        Args:
            path: Directory path containing saved index
        """
        load_path = Path(path)
        self.index = faiss.read_index(str(load_path / 'faiss.index'))
        
        with open(load_path / 'chunks.pkl', 'rb') as f:
            self.chunks = pickle.load(f)
        
        print(f" Loaded index with {self.index.ntotal} vectors")