"""Evaluation suite for RAG pipeline using RAGAS-style metrics."""

import numpy as np
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class EvalSample:
    question: str
    answer: str
    context: List[str]
    ground_truth: str


def faithfulness_score(answer: str, contexts: List[str]) -> float:
    """Measure how well the answer is grounded in retrieved context."""
    answer_words = set(answer.lower().split())
    context_words = set()
    for ctx in contexts:
        context_words.update(ctx.lower().split())

    if not answer_words:
        return 0.0
    overlap = len(answer_words & context_words)
    return min(overlap / len(answer_words), 1.0)


def answer_relevance_score(question: str, answer: str) -> float:
    """Measure how relevant the answer is to the question."""
    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())
    if not q_words:
        return 0.0
    overlap = len(q_words & a_words)
    return min(overlap / len(q_words) * 2, 1.0)


def context_precision_score(contexts: List[str], ground_truth: str) -> float:
    """Measure how precise the retrieved contexts are."""
    gt_words = set(ground_truth.lower().split())
    scores = []
    for ctx in contexts:
        ctx_words = set(ctx.lower().split())
        if gt_words:
            scores.append(len(gt_words & ctx_words) / len(gt_words))
        else:
            scores.append(0.0)
    return max(scores) if scores else 0.0


def evaluate_pipeline(samples: List[EvalSample]) -> Dict[str, float]:
    """Run full evaluation suite on a set of samples."""
    faith_scores, relevance_scores, precision_scores = [], [], []

    for sample in samples:
        faith_scores.append(faithfulness_score(sample.answer, sample.context))
        relevance_scores.append(answer_relevance_score(sample.question, sample.answer))
        precision_scores.append(context_precision_score(sample.context, sample.ground_truth))

    results = {
        'faithfulness': np.mean(faith_scores),
        'answer_relevance': np.mean(relevance_scores),
        'context_precision': np.mean(precision_scores),
        'num_samples': len(samples),
    }

    print('RAG Evaluation Results:')
    print(f'  Faithfulness:       {results["faithfulness"]:.4f}')
    print(f'  Answer Relevance:   {results["answer_relevance"]:.4f}')
    print(f'  Context Precision:  {results["context_precision"]:.4f}')
    print(f'  Samples evaluated:  {results["num_samples"]}')

    return results


def demo():
    samples = [
        EvalSample(
            question='How does RAG improve LLM responses?',
            answer='RAG improves LLM responses by retrieving relevant documents and grounding the generation in factual context, reducing hallucination.',
            context=[
                'RAG combines retrieval with generation to ground LLM responses in factual documents.',
                'Retrieval-augmented generation reduces hallucination by providing relevant context.',
            ],
            ground_truth='RAG grounds LLM responses in retrieved documents to reduce hallucination.',
        ),
        EvalSample(
            question='What is cross-encoder re-ranking?',
            answer='Cross-encoder re-ranking scores query-document pairs jointly for more precise relevance ranking than bi-encoders.',
            context=[
                'Cross-encoder models score query-document pairs for precise re-ranking.',
                'Bi-encoders encode query and document separately, cross-encoders encode them jointly.',
            ],
            ground_truth='Cross-encoders jointly encode query-document pairs for precise re-ranking.',
        ),
    ]

    evaluate_pipeline(samples)


if __name__ == '__main__':
    demo()
