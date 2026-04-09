from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PipelineTrace(BaseModel):
    sample_id: str
    question: str
    question_type: str
    entities: list[str]
    mapped_nodes: list[str]
    candidate_evidence: list[dict[str, Any]] = Field(default_factory=list)
    selected_evidence: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_errors: list[str] = Field(default_factory=list)
    final_answer: str
