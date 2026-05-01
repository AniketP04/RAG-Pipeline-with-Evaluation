"""Hybrid retrieval: dense (FAISS) + sparse (BM25) with RRF fusion."""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Document:
    content: str
    metadata: dict
    score: float = 0.0


class DenseRetriever:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = []
        self.documents = []

    def _embed(self, text: str) -> np.ndarray:
        np.random.seed(abs(hash(text)) % 2**31)
        vec = np.random.randn(self.dimension).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def add_documents(self, documents: List[str]):
        for doc in documents:
            self.index.append(self._embed(doc))
            self.documents.append(doc)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query_vec = self._embed(query)
        scores = [float(np.dot(query_vec, v)) for v in self.index]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.documents[i], s) for i, s in ranked]


class SparseRetriever:
    """BM25-style sparse retrieval (simplified)."""

    def __init__(self):
        self.documents = []
        self.doc_freqs = {}

    def add_documents(self, documents: List[str]):
        self.documents = documents
        for doc in documents:
            terms = set(doc.lower().split())
            for term in terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query_terms = set(query.lower().split())
        n = len(self.documents)
        scores = []
        for doc in self.documents:
            doc_terms = doc.lower().split()
            doc_term_set = set(doc_terms)
            score = 0
            for term in query_terms:
                if term in doc_term_set:
                    tf = doc_terms.count(term) / len(doc_terms)
                    df = self.doc_freqs.get(term, 1)
                    idf = np.log((n - df + 0.5) / (df + 0.5) + 1)
                    score += tf * idf
            scores.append(score)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.documents[i], s) for i, s in ranked]


def reciprocal_rank_fusion(
    results_list: List[List[Tuple[str, float]]],
    k: int = 60,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion."""
    scores = {}
    for results in results_list:
        for rank, (doc, _) in enumerate(results):
            scores[doc] = scores.get(doc, 0) + 1.0 / (k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked


class HybridRetriever:
    def __init__(self, dimension: int = 384):
        self.dense = DenseRetriever(dimension)
        self.sparse = SparseRetriever()

    def add_documents(self, documents: List[str]):
        self.dense.add_documents(documents)
        self.sparse.add_documents(documents)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        dense_results = self.dense.search(query, top_k=top_k * 2)
        sparse_results = self.sparse.search(query, top_k=top_k * 2)
        fused = reciprocal_rank_fusion([dense_results, sparse_results], top_k=top_k)
        return fused


if __name__ == '__main__':
    docs = [
        'RAG pipelines combine retrieval with generation for grounded responses.',
        'FAISS enables fast similarity search on dense vector embeddings.',
        'BM25 is a probabilistic ranking function used in information retrieval.',
        'Cross-encoder models score query-document pairs for precise re-ranking.',
        'Chunking strategies significantly affect RAG retrieval quality.',
    ]

    retriever = HybridRetriever()
    retriever.add_documents(docs)

    query = 'How does retrieval work in RAG?'
    results = retriever.search(query, top_k=3)
    print(f'Query: {query}\n')
    for doc, score in results:
        print(f'  [{score:.4f}] {doc}')
