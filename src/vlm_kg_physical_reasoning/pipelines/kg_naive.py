from __future__ import annotations

from vlm_kg_physical_reasoning.data.sample import Sample
from vlm_kg_physical_reasoning.extraction.entity_extraction import EntityExtractor
from vlm_kg_physical_reasoning.models.vlm_spine import VLMBackbone
from vlm_kg_physical_reasoning.retrieval.node_mapper import NodeMapper
from vlm_kg_physical_reasoning.retrieval.question_classifier import QuestionClassifier
from vlm_kg_physical_reasoning.retrieval.retriever_protocol import RetrieverProtocol
from vlm_kg_physical_reasoning.tracing.trace_builder import TraceBuilder
from vlm_kg_physical_reasoning.tracing.trace_schema import PipelineTrace


class NaiveKGPipeline:
    """Run an end-to-end KG-augmented VLM pipeline.

    The retriever is injected, so this pipeline can run either:
    - BasicRetriever for naive KG
    - QuestionAwareRetriever for improved KG retrieval
    """

    def __init__(
        self,
        vlm: VLMBackbone,
        entity_extractor: EntityExtractor,
        node_mapper: NodeMapper,
        retriever: RetrieverProtocol,
        question_classifier: QuestionClassifier,
        trace_builder: TraceBuilder,
        max_entities: int,
        max_evidence_triples: int,
    ) -> None:
        self.vlm = vlm
        self.entity_extractor = entity_extractor
        self.node_mapper = node_mapper
        self.retriever = retriever
        self.question_classifier = question_classifier
        self.trace_builder = trace_builder
        self.max_entities = max_entities
        self.max_evidence_triples = max_evidence_triples

    def run(self, sample: Sample) -> PipelineTrace:
        question_type = self.question_classifier.classify(sample.question)
        entities = self.entity_extractor.extract(sample, max_entities=self.max_entities)
        mapped_nodes = self.node_mapper.map_entities(entities)

        retrieval_result = self.retriever.retrieve(
            mapped_nodes=mapped_nodes,
            question=sample.question,
            top_k=self.max_evidence_triples,
            question_type=question_type,
        )

        evidence_strings = [
            f"{edge.subject} {edge.relation} {edge.object}"
            for edge in retrieval_result.selected_edges
        ]

        final_answer = self.vlm.answer(
            image_path=sample.image_path,
            question=sample.question,
            evidence=evidence_strings,
        )

        return self.trace_builder.build(
            sample=sample,
            question_type=question_type,
            entities=entities,
            mapped_nodes=mapped_nodes,
            candidate_evidence=retrieval_result.candidate_edges,
            selected_evidence=retrieval_result.selected_edges,
            retrieval_errors=retrieval_result.retrieval_errors,
            final_answer=final_answer,
        )

