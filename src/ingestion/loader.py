"""Document loading module for ingesting raw documents.

Loads JSON documents from directories for processing in the RAG pipeline.
"""

from pathlib import Path
from typing import List, Dict
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load documents from disk for processing.
    
    Recursively loads JSON files from a directory and provides methods
    to access document collections.
    
    Attributes:
        data_dir: Path to directory containing documents
    """
    def __init__(self, data_dir: str = "data/raw"):
        """Initialize document loader.
        
        Args:
            data_dir: Path to directory containing JSON documents
        """
        self.data_dir = Path(data_dir)
    
    def load_json(self, filepath: Path) -> Dict:
        """Load single JSON document.
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            Dict: Parsed JSON content
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_all(self) -> List[Dict]:
        """Load all JSON documents from directory recursively.
        
        Returns:
            List[Dict]: All loaded documents
        """
        documents = []
        
        for filepath in self.data_dir.rglob('*.json'):
            try:
                doc = self.load_json(filepath)
                documents.append(doc)
                
                if len(documents) % 50 == 0:
                    logger.info(f"Loaded {len(documents)} documents...")
            
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
        
        logger.info(f" Loaded {len(documents)} documents total")
        return documents

if __name__ == '__main__':
    loader = DocumentLoader('data/raw/arxiv')
    docs = loader.load_all()
    print(f"\n Sample document:")
    print(f"Title: {docs[0]['title']}")
    print(f"Length: {len(docs[0]['content'])} chars")