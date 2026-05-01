"""Chunking strategies for RAG pipeline."""

from typing import List
import re


def fixed_size_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def sentence_chunk(text: str, max_sentences: int = 5, overlap: int = 1) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    start = 0
    while start < len(sentences):
        end = min(start + max_sentences, len(sentences))
        chunk = ' '.join(sentences[start:end])
        chunks.append(chunk)
        start += max_sentences - overlap
    return chunks


def semantic_chunk(text: str, similarity_threshold: float = 0.5) -> List[str]:
    """Semantic chunking based on topic shifts (simplified)."""
    paragraphs = text.split('\n\n')
    if not paragraphs:
        return [text]

    chunks = []
    current_chunk = paragraphs[0]

    for para in paragraphs[1:]:
        word_overlap = len(set(current_chunk.lower().split()) & set(para.lower().split()))
        total_words = len(set(current_chunk.lower().split()) | set(para.lower().split()))
        similarity = word_overlap / max(total_words, 1)

        if similarity >= similarity_threshold:
            current_chunk += '\n\n' + para
        else:
            chunks.append(current_chunk)
            current_chunk = para

    chunks.append(current_chunk)
    return chunks


def compare_strategies(text: str):
    print('Chunking Strategy Comparison:')
    print(f'  Text length: {len(text.split())} words\n')

    for name, fn in [
        ('Fixed-size (500w)', lambda t: fixed_size_chunk(t, 500, 50)),
        ('Sentence (5 sent)', lambda t: sentence_chunk(t, 5, 1)),
        ('Semantic', lambda t: semantic_chunk(t, 0.3)),
    ]:
        chunks = fn(text)
        sizes = [len(c.split()) for c in chunks]
        print(f'  {name}: {len(chunks)} chunks, avg={sum(sizes)/len(sizes):.0f} words')
