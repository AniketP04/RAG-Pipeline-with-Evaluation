"""RAG system evaluation metrics using LLM-based assessments.

Provides methods to evaluate answer quality, faithfulness to context, and
retrieval precision using GPT-4 as an evaluation model.
"""

from typing import List, Dict
from openai import OpenAI
import os
import re
import time


class RAGEvaluator:
    """Evaluates RAG system responses using multiple quality metrics.
    
    Uses OpenAI's GPT model to assess answer relevancy, faithfulness to context,
    and precision of retrieved chunks.
    
    Attributes:
        client: OpenAI API client
        model: LLM model name used for evaluation
    """
    
    def __init__(self, model: str = "gpt-4.1-mini-2025-04-14"):
        """Initialize RAG evaluator with OpenAI API.
        
        Args:
            model: LLM model to use for evaluation (default: gpt-4.1-mini-2025-04-14)
        
        Raises:
            ValueError: If OPENAI_API_KEY environment variable is not set
        """
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _ask_llm(self, prompt: str, max_tokens: int = 10) -> str:
        """Send prompt to LLM and get response.
        
        Args:
            prompt: Text prompt to send to the model
            max_tokens: Maximum tokens in the response (default: 10)
        
        Returns:
            str: Stripped response text from the model
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=0
        )

        return response.choices[0].message.content.strip()

    def answer_relevancy(self, question: str, answer: str) -> float:
        """Rate how well the answer addresses the question.
        
        Args:
            question: The user's question
            answer: The generated answer
        
        Returns:
            float: Relevancy score from 0-5 (higher is better)
        """

        prompt = f"""Rate this answer's relevancy on a scale of 0-5.

Question: {question}

Answer: {answer}

Provide ONLY a number between 0 and 5.
"""

        try:
            result = self._ask_llm(prompt)

            match = re.search(r"\d+\.?\d*", result)

            if not match:
                return 0.0

            score = float(match.group())

            return min(max(score, 0), 5)

        except Exception:
            return 0.0

    def faithfulness(self, answer: str, context_chunks: List[str]) -> float:
        """Check if answer is grounded in provided context without hallucination.
        
        Args:
            answer: The generated answer
            context_chunks: List of context chunks used for generation
        
        Returns:
            float: Faithfulness score 0-1 (1 = fully grounded, 0 = contains hallucinations)
        """

        context = "\n\n".join(context_chunks[:3])  # Top 3 chunks

        prompt = f"""Does the answer contain information NOT present in the context?

Context:
{context[:2000]}

Answer:
{answer}

Reply ONLY with:
- yes
- no
"""

        try:
            result = self._ask_llm(prompt).lower()

            return 0.0 if "yes" in result else 1.0

        except Exception:
            return 0.0

    def context_precision(self, question: str, context_chunks: List[str]) -> float:
        """Calculate precision of retrieved context chunks.
        
        Measures what percentage of retrieved chunks are relevant to answering
        the question.
        
        Args:
            question: The user's question
            context_chunks: List of retrieved context chunks
        
        Returns:
            float: Precision score 0-1 (1 = all relevant, 0 = none relevant)
        """

        if not context_chunks:
            return 0.0

        relevant_count = 0

        for chunk in context_chunks:

            prompt = f"""Is this context relevant for answering the question?

Question:
{question}

Context:
{chunk[:500]}

Reply ONLY with:
- yes
- no
"""

            try:
                result = self._ask_llm(prompt).lower()

                if "yes" in result:
                    relevant_count += 1

            except Exception:
                continue

        return relevant_count / len(context_chunks)

    def evaluate_response(
        self,
        question: str,
        answer: str,
        retrieved_chunks: List[str],
        measure_latency: bool = False
    ) -> Dict:
        """Run all evaluation metrics"""

        start_time = time.time() if measure_latency else None

        print(f"  Evaluating: {question[:50]}...")

        metrics = {
            "answer_relevancy": self.answer_relevancy(question, answer),
            "faithfulness": self.faithfulness(answer, retrieved_chunks),
            "context_precision": self.context_precision(
                question,
                retrieved_chunks[:5]
            )
        }

        # Weighted overall score
        metrics["overall_score"] = (
            (metrics["answer_relevancy"] / 5) * 0.4 +
            metrics["faithfulness"] * 0.3 +
            metrics["context_precision"] * 0.3
        )

        if measure_latency:
            metrics["evaluation_time_s"] = (
                time.time() - start_time
            )

        return metrics