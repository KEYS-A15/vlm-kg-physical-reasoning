from __future__ import annotations

from vlm_kg_physical_reasoning.data.sample import Sample
from vlm_kg_physical_reasoning.retrieval.conceptnet_client import ConceptNetEdge
from vlm_kg_physical_reasoning.tracing.trace_schema import PipelineTrace


class TraceBuilder:
    """Build a serializable trace for KG-augmented runs."""

    def build(
        self,
        sample: Sample,
        question_type: str,
        entities: list[str],
        mapped_nodes: list[str],
        candidate_evidence: list[ConceptNetEdge],
        selected_evidence: list[ConceptNetEdge],
        retrieval_errors: list[str],
        final_answer: str,
    ) -> PipelineTrace:
        return PipelineTrace(
            sample_id=sample.sample_id,
            question=sample.question,
            question_type=question_type,
            entities=entities,
            mapped_nodes=mapped_nodes,
            candidate_evidence=[edge.model_dump() for edge in candidate_evidence],
            selected_evidence=[edge.model_dump() for edge in selected_evidence],
            retrieval_errors=retrieval_errors,
            final_answer=final_answer,
        )
