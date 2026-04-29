from __future__ import annotations

from vlm_kg_physical_reasoning.retrieval.conceptnet_client import ConceptNetEdge
from vlm_kg_physical_reasoning.retrieval.question_aware_retriever import (
    QuestionAwareRetriever,
)


class FakeConceptNetClient:
    def fetch_edges(self, node_uri: str, limit: int | None = None) -> list[ConceptNetEdge]:
        return [
            ConceptNetEdge(
                subject="books",
                relation="AtLocation",
                object="desk",
                weight=8.0,
                uri="/a/1",
            ),
            ConceptNetEdge(
                subject="books",
                relation="ReceivesAction",
                object="read",
                weight=10.0,
                uri="/a/2",
            ),
            ConceptNetEdge(
                subject="cup",
                relation="UsedFor",
                object="drinking",
                weight=3.0,
                uri="/a/3",
            ),
            ConceptNetEdge(
                subject="cup",
                relation="HasContext",
                object="clothing",
                weight=20.0,
                uri="/a/4",
            ),
        ]


def test_spatial_question_suppresses_kg_evidence() -> None:
    retriever = QuestionAwareRetriever(
        client=FakeConceptNetClient(),
        max_edges_per_node=10,
        suppress_spatial_kg=True,
    )

    result = retriever.retrieve(
        mapped_nodes=["/c/en/cup", "/c/en/books"],
        question="Where is the cup relative to the books?",
        question_type="spatial",
        top_k=5,
    )

    assert len(result.candidate_edges) > 0
    assert result.selected_edges == []


def test_affordance_question_prefers_usedfor_and_removes_noisy_edges() -> None:
    retriever = QuestionAwareRetriever(
        client=FakeConceptNetClient(),
        max_edges_per_node=10,
        suppress_spatial_kg=False,
    )

    result = retriever.retrieve(
        mapped_nodes=["/c/en/cup"],
        question="Which object is used for drinking?",
        question_type="affordance",
        top_k=3,
    )

    selected_relations = [edge.relation.lower() for edge in result.selected_edges]

    assert "usedfor" in selected_relations
    assert "hascontext" not in selected_relations
    assert "receivesaction" in selected_relations or "usedfor" in selected_relations


def test_unknown_question_type_falls_back_to_physical_general() -> None:
    retriever = QuestionAwareRetriever(
        client=FakeConceptNetClient(),
        max_edges_per_node=10,
        suppress_spatial_kg=False,
    )

    result = retriever.retrieve(
        mapped_nodes=["/c/en/cup"],
        question="What is shown?",
        question_type="unknown_type",
        top_k=5,
    )

    selected_relations = {edge.relation.lower() for edge in result.selected_edges}

    assert "hascontext" not in selected_relations