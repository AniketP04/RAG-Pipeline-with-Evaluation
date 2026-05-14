"""Pydantic data models for RAG API communication.

Defines request/response models for type validation and API documentation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class QueryRequest(BaseModel):
    """User query request for the RAG system.
    
    Attributes:
        question: The user's question to retrieve context and generate answers for
        top_k: Number of context chunks to retrieve (1-20, default: 5)
    """
    question: str = Field(..., description="The question to answer")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")

class Source(BaseModel):
    """Retrieved source chunk with relevance metadata.
    
    Attributes:
        chunk_id: Unique identifier for the retrieved chunk
        doc_title: Title of the source document
        content: The relevant text content from the chunk
        score: Relevance score (0-1) indicating how relevant the chunk is
    """
    chunk_id: str
    doc_title: str
    content: str
    score: float

class QueryResponse(BaseModel):
    """Complete response to a RAG query.
    
    Attributes:
        question: Echo of the user's original question
        answer: Generated answer based on retrieved context
        sources: List of source chunks used to generate the answer
        metadata: Additional metadata about the query processing
    """
    question: str
    answer: str
    sources: List[Source]
    metadata: Dict = Field(default_factory=dict)

class HealthResponse(BaseModel):
    """System health status information.
    
    Attributes:
        status: Current system status ('healthy', 'degraded', 'down')
        version: API version number
        index_loaded: Whether search index is loaded and ready
        total_chunks: Total number of indexed chunks available for retrieval
    """
    status: str
    version: str
    index_loaded: bool
    total_chunks: int

class StatsResponse(BaseModel):
    """Aggregated system performance statistics.
    
    Attributes:
        total_queries: Total number of queries processed
        avg_latency_ms: Average response time in milliseconds
        total_tokens_used: Total tokens consumed by all queries
        avg_tokens_per_query: Average tokens used per query
    """
    total_queries: int
    avg_latency_ms: float
    total_tokens_used: int
    avg_tokens_per_query: float