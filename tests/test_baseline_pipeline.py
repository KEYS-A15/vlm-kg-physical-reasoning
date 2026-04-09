from __future__ import annotations

from vlm_kg_physical_reasoning.data.sample import Sample
from vlm_kg_physical_reasoning.pipelines.baseline import BaselinePipeline


class FakeVLM:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, list[str] | None]] = []

    def answer(self, image_path: str, question: str, evidence: list[str] | None = None) -> str:
        self.calls.append((image_path, question, evidence))
        return "baseline answer"

    def extract_entities(self, image_path: str, question: str, max_entities: int) -> list[str]:
        return ["cup"]


def test_baseline_pipeline_wires_sample_into_vlm() -> None:
    vlm = FakeVLM()
    pipeline = BaselinePipeline(vlm=vlm)
    sample = Sample(sample_id="s1", image_path="image.png", question="What is on the table?")

    result = pipeline.run(sample)

    assert result["sample_id"] == "s1"
    assert result["final_answer"] == "baseline answer"
    assert vlm.calls == [("image.png", "What is on the table?", None)]
