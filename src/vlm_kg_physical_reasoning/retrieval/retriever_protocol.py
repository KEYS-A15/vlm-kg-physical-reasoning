from __future__ import annotations

from typing import Protocol

from vlm_kg_physical_reasoning.retrieval.basic_retriever import RetrievalResult


class RetrieverProtocol(Protocol):
    def retrieve(
        self,
        mapped_nodes: list[str],
        question: str,
        top_k: int = 5,
        question_type: str | None = None,
    ) -> RetrievalResult:
        ...