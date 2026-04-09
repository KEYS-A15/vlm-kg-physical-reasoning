from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Sample(BaseModel):
    sample_id: str
    image_path: str
    question: str
    gold_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)