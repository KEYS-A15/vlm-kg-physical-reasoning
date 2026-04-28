from __future__ import annotations

from pathlib import Path

from vlm_kg_physical_reasoning.data.sample import Sample
from vlm_kg_physical_reasoning.models.vlm_spine import VLMBackbone
from vlm_kg_physical_reasoning.pipelines.results import BaselinePrediction


class BaselinePipeline:
    """Run the direct VLM baseline without KG evidence.

    This pipeline is intentionally simple: image + question only.
    It is the reference point for later KG-augmented comparisons.
    """

    def __init__(self, vlm: VLMBackbone) -> None:
        self.vlm = vlm

    def run(self, sample: Sample) -> BaselinePrediction:
        image_path = Path(sample.image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Baseline image not found for sample '{sample.sample_id}': {sample.image_path}"
            )

        final_answer = self.vlm.answer(str(image_path), sample.question).strip()

        if not final_answer:
            final_answer = "[EMPTY_MODEL_RESPONSE]"

        return BaselinePrediction(
            sample_id=sample.sample_id,
            image_path=str(image_path),
            question=sample.question,
            gold_answer=sample.gold_answer,
            final_answer=final_answer,
            metadata={
                "has_gold_answer": sample.gold_answer is not None,
            },
        )