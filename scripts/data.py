"""Script to download and prepare ArXiv dataset for RAG pipeline.

Downloads 200 ArXiv papers from the HuggingFace datasets library and
converts them to individual JSON files for document ingestion.
"""

from datasets import load_dataset
from pathlib import Path
import json


print("Downloading ArXiv dataset...")

dataset = load_dataset("ccdv/arxiv-summarization", split="train[:200]")


output_dir = Path("data/raw/arxiv")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Saving {len(dataset)} documents...")

for idx, item in enumerate(dataset):
    doc = {
        'id': f'arxiv_{idx}',
        'title': item.get('article', '').split('\n')[0][:100],
        'content': item.get('article', ''),
        'abstract': item.get('abstract', '')
    }
    
    with open(output_dir / f'doc_{idx}.json', 'w') as f:
        json.dump(doc, f, indent=2)
    
    if (idx + 1) % 50 == 0:
        print(f"  Saved {idx + 1}/{len(dataset)}...")

print(f"Created {len(dataset)} documents")