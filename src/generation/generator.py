"""Answer generation module for RAG system.

Generates natural language answers based on retrieved context chunks,
using OpenAI's GPT models.
"""

import os
from typing import List, Dict, Tuple
from openai import OpenAI

class AnswerGenerator:
    """Generate answers from context using LLM.
    
    Combines retrieved context chunks with user questions to generate
    coherent, cited answers using OpenAI's GPT models.
    
    Attributes:
        client: OpenAI API client
        model: LLM model name for generation
        system_prompt: Instructions for the model
    """
    def __init__(self, model: str = "gpt-4.1-mini-2025-04-14"):
        """Initialize answer generator.
        
        Args:
            model: LLM model to use (default: gpt-4.1-mini-2025-04-14)
        
        Raises:
            ValueError: If OPENAI_API_KEY environment variable not found
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")

        self.client = OpenAI(api_key=api_key)
        self.model = model

        self.system_prompt = """You are a helpful AI assistant that answers questions based on provided context.

Rules:
1. Only use information from the provided context
2. If the answer is not in the context, say "I don't have enough information"
3. Cite sources using [1], [2], etc.
4. Be concise but complete
5. Do not make up information
"""

    def _format_context(self, chunks: List[Tuple[Dict, float]]) -> str:
        """Format retrieved chunks as numbered context for LLM.
        
        Args:
            chunks: List of (chunk_dict, score) tuples
        
        Returns:
            str: Formatted context string with numbered references
        """
        context_parts = []

        for i, (chunk, score) in enumerate(chunks, 1):
            context_parts.append(f"[{i}] {chunk['content']}\n")

        return "\n".join(context_parts)

    def generate(
        self,
        question: str,
        retrieved_chunks: List[Tuple[Dict, float]],
        max_tokens: int = 500
    ) -> Dict:
        """Generate answer from question and retrieved context.
        
        Args:
            question: User's question
            retrieved_chunks: List of (chunk_dict, relevance_score) tuples
            max_tokens: Maximum tokens in generated answer (default: 500)
        
        Returns:
            Dict: Contains 'question', 'answer', 'sources', 'model', 'tokens_used'
        """

        context = self._format_context(retrieved_chunks)

        user_prompt = f"""Context:
{context}

Question: {question}

Answer using ONLY the context above. Include citations [1], [2], etc.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0
            )

            answer = response.choices[0].message.content

            sources = [
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_title": chunk["doc_title"],
                    "content": chunk["content"][:200] + "...",
                    "score": score
                }
                for chunk, score in retrieved_chunks
            ]

            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "model": self.model,
                "tokens_used": response.usage.total_tokens
            }

        except Exception as e:
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "sources": [],
                "model": self.model,
                "tokens_used": 0
            }