from __future__ import annotations

from vlm_kg_physical_reasoning.data.sample import Sample
from vlm_kg_physical_reasoning.models.vlm_spine import VLMBackbone


class BaselinePipeline:
    """Run the direct VLM baseline without KG evidence."""

    def __init__(self, vlm: VLMBackbone) -> None:
        self.vlm = vlm

    def run(self, sample: Sample) -> dict[str, str | None]:
        final_answer = self.vlm.answer(sample.image_path, sample.question)
        return {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "gold_answer": sample.gold_answer,
            "final_answer": final_answer,
        }
