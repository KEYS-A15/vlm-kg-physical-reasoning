from __future__ import annotations

from vlm_kg_physical_reasoning.data.sample import Sample
from vlm_kg_physical_reasoning.models.vlm_spine import VLMBackbone


class EntityExtractor:
    """Thin adapter that delegates entity extraction to the VLM wrapper."""

    def __init__(self, vlm: VLMBackbone) -> None:
        self.vlm = vlm

    def extract(self, sample: Sample, max_entities: int = 5) -> list[str]:
        return self.vlm.extract_entities(
            image_path=sample.image_path,
            question=sample.question,
            max_entities=max_entities,
        )
