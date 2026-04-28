from __future__ import annotations

from pathlib import Path

import pytest

from vlm_kg_physical_reasoning.data.sample import Sample
from vlm_kg_physical_reasoning.pipelines.baseline import BaselinePipeline
from vlm_kg_physical_reasoning.pipelines.results import BaselinePrediction


class FakeVLM:
    def answer(
        self,
        image_path: str,
        question: str,
        evidence: list[str] | None = None,
    ) -> str:
        return "the cup is behind the books"

    def extract_entities(
        self,
        image_path: str,
        question: str,
        max_entities: int,
    ) -> list[str]:
        return ["cup", "books"][:max_entities]


def test_baseline_pipeline_returns_prediction_contract(tmp_path: Path) -> None:
    image = tmp_path / "demo.jpg"
    image.write_bytes(b"fake-image-bytes")

    sample = Sample(
        sample_id="demo_001",
        image_path=str(image),
        question="Where is the cup relative to the books?",
        gold_answer="behind the books",
    )

    pipeline = BaselinePipeline(vlm=FakeVLM())
    result = pipeline.run(sample)

    assert isinstance(result, BaselinePrediction)
    assert result.sample_id == "demo_001"
    assert result.image_path == str(image)
    assert result.question == "Where is the cup relative to the books?"
    assert result.gold_answer == "behind the books"
    assert result.final_answer == "the cup is behind the books"
    assert result.pipeline == "baseline_vlm"
    assert result.metadata["has_gold_answer"] is True


def test_baseline_pipeline_rejects_missing_image() -> None:
    sample = Sample(
        sample_id="missing_image",
        image_path="does/not/exist.jpg",
        question="Where is the cup?",
        gold_answer=None,
    )

    pipeline = BaselinePipeline(vlm=FakeVLM())

    with pytest.raises(FileNotFoundError):
        pipeline.run(sample)


class EmptyAnswerVLM(FakeVLM):
    def answer(
        self,
        image_path: str,
        question: str,
        evidence: list[str] | None = None,
    ) -> str:
        return "   "


def test_baseline_pipeline_handles_empty_model_response(tmp_path: Path) -> None:
    image = tmp_path / "demo.jpg"
    image.write_bytes(b"fake-image-bytes")

    sample = Sample(
        sample_id="empty_answer",
        image_path=str(image),
        question="What is shown?",
    )

    pipeline = BaselinePipeline(vlm=EmptyAnswerVLM())
    result = pipeline.run(sample)

    assert result.final_answer == "[EMPTY_MODEL_RESPONSE]"