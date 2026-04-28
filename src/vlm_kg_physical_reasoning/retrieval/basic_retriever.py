from __future__ import annotations

from dataclasses import dataclass
import re

from vlm_kg_physical_reasoning.retrieval.conceptnet_client import (
    ConceptNetClientError,
    ConceptNetClientProtocol,
    ConceptNetEdge,
)


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "where",
    "why",
    "with",
}


@dataclass(slots=True)
class RetrievalResult:
    candidate_edges: list[ConceptNetEdge]
    selected_edges: list[ConceptNetEdge]
    retrieval_errors: list[str]


class BasicRetriever:
    """Naive 1-hop retriever using ConceptNet edge weights and lexical overlap."""

    def __init__(self, client: ConceptNetClientProtocol, max_edges_per_node: int, overlap_weight: float) -> None:
        self.client = client
        self.max_edges_per_node = max_edges_per_node
        self.overlap_weight = overlap_weight

    def retrieve(
        self,
        mapped_nodes: list[str],
        question: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        all_edges: list[ConceptNetEdge] = []
        errors: list[str] = []

        for node_uri in mapped_nodes:
            try:
                node_edges = self.client.fetch_edges(node_uri, limit=self.max_edges_per_node)
            except ConceptNetClientError as exc:
                errors.append(f"{node_uri}: {exc}")
                continue

            all_edges.extend(node_edges)

        deduped_edges = self._dedupe_edges(all_edges)
        ranked_edges = sorted(
            deduped_edges,
            key=lambda edge: self._score_edge(edge, question),
            reverse=True,
        )

        return RetrievalResult(
            candidate_edges=deduped_edges,
            selected_edges=ranked_edges[:top_k],
            retrieval_errors=errors,
        )

    @staticmethod
    def _dedupe_edges(edges: list[ConceptNetEdge]) -> list[ConceptNetEdge]:
        seen: set[tuple[str, str, str]] = set()
        deduped: list[ConceptNetEdge] = []

        for edge in edges:
            key = (edge.subject, edge.relation, edge.object)
            if key in seen:
                continue

            seen.add(key)
            deduped.append(edge)

        return deduped

    def _score_edge(self, edge: ConceptNetEdge, question: str) -> float:
        question_tokens = self._tokenize(question)
        edge_tokens = self._tokenize(f"{edge.subject} {edge.relation} {edge.object}")
        overlap = len(question_tokens & edge_tokens)
        return ((1 - self.overlap_weight) * edge.weight) + (self.overlap_weight * overlap)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {
            token
            for token in _TOKEN_PATTERN.findall(text.lower())
            if token not in _STOPWORDS and len(token) > 1
        }
