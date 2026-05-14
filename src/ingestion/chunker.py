"""Document chunking module for text preprocessing.

Splits large documents into overlapping chunks for embedding and retrieval.
"""

from typing import List, Dict
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DocumentChunker:
    """Split documents into overlapping chunks for embedding.
    
    Uses sliding window chunking with configurable size and overlap.
    Cleans text and generates chunk metadata.
    
    Attributes:
        chunk_size: Size of each chunk in characters (default: 1024)
        chunk_overlap: Overlap between consecutive chunks (default: 128)
    """
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 128):
        """Initialize document chunker with parameters.
        
        Args:
            chunk_size: Size of chunks in characters (default: 1024)
            chunk_overlap: Overlap between chunks (default: 128)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def clean_text(self, text: str) -> str:
        """Clean text by normalizing whitespace.
        
        Args:
            text: Raw text to clean
        
        Returns:
            str: Cleaned text
        """
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
        
        Returns:
            List[str]: List of text chunks
        """
        text = self.clean_text(text)
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            if chunk:
                chunks.append(chunk)
            
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def chunk_document(self, document: Dict) -> List[Dict]:
        """Chunk a single document and add metadata.
        
        Args:
            document: Document dict with 'content', 'id', 'title' keys
        
        Returns:
            List[Dict]: Chunks with metadata including chunk_id, doc_id, content
        """
        content = document.get('content', '')
        text_chunks = self.chunk_text(content)
        
        chunk_docs = []
        for idx, chunk_text in enumerate(text_chunks):
            chunk_doc = {
                'chunk_id': f"{document['id']}_chunk_{idx}",
                'doc_id': document['id'],
                'doc_title': document.get('title', ''),
                'chunk_index': idx,
                'content': chunk_text,
                'metadata': {
                    'total_chunks': len(text_chunks),
                    'char_count': len(chunk_text),
                    'word_count': len(chunk_text.split())
                }
            }
            chunk_docs.append(chunk_doc)
        
        return chunk_docs
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Chunk multiple documents.
        
        Args:
            documents: List of document dicts
        
        Returns:
            List[Dict]: All chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        logger.info(f" Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks

if __name__ == '__main__':
    from loader import DocumentLoader
    import json
    
    loader = DocumentLoader('data/raw/arxiv')
    docs = loader.load_all()
    
    chunker = DocumentChunker(chunk_size=1024, chunk_overlap=128)
    chunks = chunker.chunk_documents(docs)
    
    output_path = Path('data/processed/chunks.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(chunks, f, indent=2)
    
    print(f" Saved {len(chunks)} chunks to {output_path}")