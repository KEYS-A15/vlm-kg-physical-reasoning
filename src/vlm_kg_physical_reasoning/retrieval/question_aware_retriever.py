from __future__ import annotations

from vlm_kg_physical_reasoning.retrieval.basic_retriever import (
    BasicRetriever,
    RetrievalResult,
)
from vlm_kg_physical_reasoning.retrieval.conceptnet_client import (
    ConceptNetClientProtocol,
    ConceptNetEdge,
)


NOISY_RELATIONS: set[str] = {
    "relatedto",
    "hascontext",
    "synonym",
    "antonym",
    "distinctfrom",
    "etymologicallyrelatedto",
    "derivedfrom",
    "formof",
}


QUESTION_RELATION_PRIORS: dict[str, dict[str, float]] = {
    "spatial": {
        # For image-relative spatial questions, KG is often risky.
        # Keep only weak/general support relations if they survive.
        "atlocation": 0.4,
        "locatednear": 0.3,
        "partof": 0.2,
    },
    "affordance": {
        "usedfor": 2.0,
        "capableof": 1.8,
        "receivesaction": 0.8,
    },
    "property": {
        "hasproperty": 2.0,
        "madeof": 1.8,
        "hasa": 1.2,
        "partof": 0.8,
    },
    "causal": {
        "causes": 2.0,
        "causesdesire": 1.5,
        "hassubevent": 1.2,
        "motivatedbygoal": 1.0,
    },
    "physical_general": {
        "usedfor": 1.0,
        "capableof": 1.0,
        "hasproperty": 1.0,
        "madeof": 1.0,
        "atlocation": 0.6,
        "hasa": 0.6,
        "partof": 0.6,
    },
}


class QuestionAwareRetriever(BasicRetriever):
    """Question-aware ConceptNet retriever.

    This retriever is the first retrieval-improvement module.

    It keeps the same external interface as BasicRetriever but changes selection by:
    1. removing noisy/generic relations,
    2. applying relation priors based on question type,
    3. optionally suppressing KG evidence for image-relative spatial questions.

    BasicRetriever remains the naive baseline.
    """

    def __init__(
        self,
        client: ConceptNetClientProtocol,
        max_edges_per_node: int,
        overlap_weight: float = 0.35,
        relation_prior_weight: float = 1.0,
        suppress_spatial_kg: bool = True,
    ) -> None:
        super().__init__(
            client=client,
            max_edges_per_node=max_edges_per_node,
            overlap_weight=overlap_weight,
        )
        self.relation_prior_weight = relation_prior_weight
        self.suppress_spatial_kg = suppress_spatial_kg

    def retrieve(
        self,
        mapped_nodes: list[str],
        question: str,
        top_k: int = 5,
        question_type: str = "physical_general",
    ) -> RetrievalResult:
        base_result = super().retrieve(
            mapped_nodes=mapped_nodes,
            question=question,
            top_k=max(top_k * 4, top_k),
        )

        normalized_question_type = self._normalize_question_type(question_type)

        # For image-relative spatial questions, ConceptNet usually cannot answer
        # left/right/front/between. Returning no evidence avoids distracting the VLM.
        if self.suppress_spatial_kg and normalized_question_type == "spatial":
            return RetrievalResult(
                candidate_edges=base_result.candidate_edges,
                selected_edges=[],
                retrieval_errors=base_result.retrieval_errors,
            )

        filtered_edges = [
            edge
            for edge in base_result.candidate_edges
            if self._is_allowed_edge(edge=edge, question_type=normalized_question_type)
        ]

        ranked_edges = sorted(
            filtered_edges,
            key=lambda edge: self._score_question_aware_edge(
                edge=edge,
                question=question,
                question_type=normalized_question_type,
            ),
            reverse=True,
        )

        return RetrievalResult(
            candidate_edges=base_result.candidate_edges,
            selected_edges=ranked_edges[:top_k],
            retrieval_errors=base_result.retrieval_errors,
        )

    def _is_allowed_edge(self, edge: ConceptNetEdge, question_type: str) -> bool:
        relation = self._normalize_relation(edge.relation)

        if relation in NOISY_RELATIONS:
            return False

        priors = QUESTION_RELATION_PRIORS.get(
            question_type,
            QUESTION_RELATION_PRIORS["physical_general"],
        )

        return relation in priors

    def _score_question_aware_edge(
        self,
        edge: ConceptNetEdge,
        question: str,
        question_type: str,
    ) -> float:
        base_score = self._score_edge(edge=edge, question=question)
        relation = self._normalize_relation(edge.relation)

        priors = QUESTION_RELATION_PRIORS.get(
            question_type,
            QUESTION_RELATION_PRIORS["physical_general"],
        )
        relation_prior = priors.get(relation, 0.0)

        return base_score + (self.relation_prior_weight * relation_prior)

    @staticmethod
    def _normalize_question_type(question_type: str) -> str:
        cleaned = question_type.strip().lower()
        if cleaned in QUESTION_RELATION_PRIORS:
            return cleaned
        return "physical_general"

    @staticmethod
    def _normalize_relation(relation: str) -> str:
        return relation.strip().lower().replace("_", "")