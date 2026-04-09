from __future__ import annotations

from vlm_kg_physical_reasoning.data.sample import Sample
from vlm_kg_physical_reasoning.extraction.entity_extraction import EntityExtractor
from vlm_kg_physical_reasoning.pipelines.kg_naive import NaiveKGPipeline
from vlm_kg_physical_reasoning.retrieval.basic_retriever import BasicRetriever
from vlm_kg_physical_reasoning.retrieval.conceptnet_client import ConceptNetEdge
from vlm_kg_physical_reasoning.retrieval.node_mapper import NodeMapper
from vlm_kg_physical_reasoning.retrieval.question_classifier import QuestionClassifier
from vlm_kg_physical_reasoning.tracing.trace_builder import TraceBuilder


class FakeVLM:
    def __init__(self) -> None:
        self.answer_calls: list[tuple[str, str, list[str] | None]] = []
        self.extract_calls: list[tuple[str, str, int]] = []

    def answer(self, image_path: str, question: str, evidence: list[str] | None = None) -> str:
        self.answer_calls.append((image_path, question, evidence))
        return "kg answer"

    def extract_entities(self, image_path: str, question: str, max_entities: int) -> list[str]:
        self.extract_calls.append((image_path, question, max_entities))
        return ["cup", "table"]


class FakeConceptNetClient:
    def fetch_edges(self, node_uri: str, limit: int | None = None) -> list[ConceptNetEdge]:
        if node_uri.endswith("cup"):
            return [
                ConceptNetEdge(
                    subject="cup",
                    relation="atlocation",
                    object="table",
                    weight=2.0,
                    uri="/a/1",
                )
            ]
        return [
            ConceptNetEdge(
                subject="table",
                relation="usedfor",
                object="holding objects",
                weight=1.0,
                uri="/a/2",
            )
        ]


def test_naive_kg_pipeline_wires_extraction_retrieval_and_trace() -> None:
    vlm = FakeVLM()
    pipeline = NaiveKGPipeline(
        vlm=vlm,
        entity_extractor=EntityExtractor(vlm),
        node_mapper=NodeMapper(),
        retriever=BasicRetriever(client=FakeConceptNetClient(), max_edges_per_node=5),
        question_classifier=QuestionClassifier(),
        trace_builder=TraceBuilder(),
        max_entities=3,
        max_evidence_triples=1,
    )
    sample = Sample(
        sample_id="s1",
        image_path="image.png",
        question="What happens if the cup is on the table?",
    )

    trace = pipeline.run(sample)

    assert trace.sample_id == "s1"
    assert trace.entities == ["cup", "table"]
    assert trace.mapped_nodes == ["/c/en/cup", "/c/en/table"]
    assert len(trace.candidate_evidence) == 2
    assert len(trace.selected_evidence) == 1
    assert trace.final_answer == "kg answer"
    assert vlm.extract_calls == [("image.png", sample.question, 3)]
    assert vlm.answer_calls == [
        ("image.png", sample.question, ["cup atlocation table"])
    ]
