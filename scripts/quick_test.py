"""Quick test of RAG system components.

Tests retrieval, generation, and evaluation components on sample questions.
Provides performance metrics and example outputs.
"""

import sys
sys.path.append('src')

from retrieval.hybrid import HybridSearch
from generation.generator import AnswerGenerator
from evaluation.metrics import RAGEvaluator

print("Quick System Test\n")

# Load search
print("Loading search index...")
hybrid = HybridSearch()
hybrid.load('data/embeddings/hybrid_index')

# Create components
print("Initializing generator...")
generator = AnswerGenerator()

print("Initializing evaluator...")
evaluator = RAGEvaluator()

# Test questions
questions = [
    "What is machine learning?",
    "How do neural networks work?",
    "What is deep learning?"
]

print("\n" + "="*60)
for i, question in enumerate(questions, 1):
    print(f"\n[{i}/{len(questions)}] {question}")
    print("-"*60)
    
    # Retrieve
    chunks = hybrid.search(question, k=5)
    print(f" Retrieved {len(chunks)} chunks")
    
    # Generate
    result = generator.generate(question, chunks)
    print(f"\n Answer:\n{result['answer'][:200]}...")
    print(f"\n Tokens: {result['tokens_used']}")
    
    # Evaluate
    context = [c['content'] for c, _ in chunks]
    metrics = evaluator.evaluate_response(question, result['answer'], context)
    
    print(f"\n Metrics:")
    print(f" Relevancy: {metrics['answer_relevancy']:.1f}/5")
    print(f" Faithfulness: {metrics['faithfulness']:.0%}")
    print(f" Overall: {metrics['overall_score']:.0%}")

print("\n" + "="*60)
print("System test complete!")