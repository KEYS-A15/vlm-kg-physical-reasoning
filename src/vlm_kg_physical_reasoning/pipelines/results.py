from __future__ import annotations

from pydantic import BaseModel, Field


class BaselinePrediction(BaseModel):
    """Serializable output for the direct VLM baseline."""

    sample_id: str
    image_path: str
    question: str
    gold_answer: str | None = None
    final_answer: str
    pipeline: str = "baseline_vlm"
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)