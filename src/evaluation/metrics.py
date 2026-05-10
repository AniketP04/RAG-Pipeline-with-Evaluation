from typing import List, Dict
from openai import OpenAI
import os
import re
import time


class RAGEvaluator:
    def __init__(self, model: str = "gpt-4.1-mini-2025-04-14"):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _ask_llm(self, prompt: str, max_tokens: int = 10) -> str:
        """Helper function for LLM calls"""

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
        """Rate answer relevancy 0-5"""

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
        """Check if answer is grounded in context (0-1)"""

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
        """What % of retrieved chunks are relevant? (0-1)"""

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