from __future__ import annotations

from pathlib import Path

import pytest

from vlm_kg_physical_reasoning.data.dataset import DemoDataset
from vlm_kg_physical_reasoning.data.sample import Sample
from vlm_kg_physical_reasoning.extraction.entity_extraction import EntityExtractor
from vlm_kg_physical_reasoning.models.qwen_vl import QwenVLModel

_SAMPLES_JSON = Path(__file__).parent.parent / "src/vlm_kg_physical_reasoning/data/samples/sample.json"
_RAW_DIR = Path(__file__).parent.parent / "src/vlm_kg_physical_reasoning/data/raw"


class CallRecordingFakeVLM:
    """Records calls and returns a fixed entity list."""

    def __init__(self, entities: list[str]) -> None:
        self._entities = entities
        self.extract_calls: list[tuple[str, str, int]] = []

    def answer(self, image_path: str, question: str, evidence: list[str] | None = None) -> str:
        return "answer"

    def extract_entities(self, image_path: str, question: str, max_entities: int) -> list[str]:
        self.extract_calls.append((image_path, question, max_entities))
        return self._entities[:]


class ParsedFakeVLM:
    """Simulates real VLM by running controlled JSON through QwenVLModel parsing logic."""

    def __init__(self, raw_json: str) -> None:
        self._raw = raw_json
        self._parser = QwenVLModel(model_name="dummy")
        self.extract_calls: list[tuple[str, str, int]] = []

    def answer(self, image_path: str, question: str, evidence: list[str] | None = None) -> str:
        return "answer"

    def extract_entities(self, image_path: str, question: str, max_entities: int) -> list[str]:
        self.extract_calls.append((image_path, question, max_entities))
        entities = self._parser._parse_entity_list(self._raw, max_entities)
        if entities:
            return entities
        return self._parser._fallback_entities(question, max_entities)


def _make_sample(question: str = "What is on the table?", image_path: str = "img.png") -> Sample:
    return Sample(sample_id="s1", image_path=image_path, question=question)


def _parser() -> QwenVLModel:
    return QwenVLModel(model_name="dummy")


# Integration tests - real images + sample.json, FakeVLM

def test_integration_sample_json_loads_all_entries() -> None:
    samples = DemoDataset(_SAMPLES_JSON).load()
    assert len(samples) == 6
    for s in samples:
        assert s.question == "Where is the cup relative to the books?"
        assert s.gold_answer == "behind the books"


def test_integration_sample_image_paths_exist() -> None:
    samples = DemoDataset(_SAMPLES_JSON).load()
    for s in samples:
        assert Path(s.image_path).exists(), f"Missing image: {s.image_path}"


def test_integration_extractor_runs_on_all_samples() -> None:
    samples = DemoDataset(_SAMPLES_JSON).load()
    vlm = CallRecordingFakeVLM(["cup", "book"])
    extractor = EntityExtractor(vlm)

    for s in samples:
        result = extractor.extract(s, max_entities=5)
        assert result == ["cup", "book"]

    assert len(vlm.extract_calls) == 6


def test_integration_extractor_forwards_real_image_path() -> None:
    samples = DemoDataset(_SAMPLES_JSON).load()
    vlm = CallRecordingFakeVLM(["cup"])
    extractor = EntityExtractor(vlm)

    extractor.extract(samples[0])

    recorded_path = vlm.extract_calls[0][0]
    assert Path(recorded_path).exists()
    assert recorded_path.endswith(".jpg")


def test_integration_raw_images_are_readable() -> None:
    Image = pytest.importorskip("PIL.Image", reason="Pillow not installed")
    for img_file in sorted(_RAW_DIR.glob("*.jpg")):
        img = Image.open(img_file)
        assert img.size[0] > 0 and img.size[1] > 0, f"Bad image: {img_file}"


def test_integration_parse_on_real_question() -> None:
    # question from sample.json: "Where is the cup relative to the books?"
    simulated_output = '["cup", "books"]'
    result = _parser()._parse_entity_list(simulated_output, max_entities=5)
    assert result == ["cup", "books"]


def test_integration_fallback_on_real_question() -> None:
    samples = DemoDataset(_SAMPLES_JSON).load()
    result = _parser()._fallback_entities(samples[0].question, max_entities=5)
    assert "cup" in result
    assert "books" in result


# Live Qwen model tests
# Skipped unless RUN_LIVE_VLM=1 is set in the environment so they never
# block CI or normal test runs.
#
# Run with:
#   RUN_LIVE_VLM=1 uv run pytest tests/test_entity_extraction.py -v -k live

_live_vlm = pytest.mark.skipif(
    not __import__("os").environ.get("RUN_LIVE_VLM"),
    reason="RUN_LIVE_VLM not set — skipping tests that load the local Qwen model",
)


@_live_vlm
def test_live_qwen_extract_entities_simple_scene() -> None:
    """Single-object scene: expects at least 'cup' or 'book' detected."""
    samples = DemoDataset(_SAMPLES_JSON).load()
    sample = samples[0]  # c1.jpg

    model = QwenVLModel(model_name="Qwen/Qwen2.5-VL-7B-Instruct")
    extractor = EntityExtractor(model)
    result = extractor.extract(sample, max_entities=5)

    assert isinstance(result, list)
    assert 1 <= len(result) <= 5
    assert all(isinstance(e, str) and e for e in result)
    assert any("cup" in e or "book" in e or "mug" in e for e in result)


@_live_vlm
def test_live_qwen_extract_entities_all_six_images() -> None:
    """Runs extraction on all 6 raw images — verifies stability across samples."""
    samples = DemoDataset(_SAMPLES_JSON).load()
    model = QwenVLModel(model_name="Qwen/Qwen2.5-VL-7B-Instruct")
    extractor = EntityExtractor(model)

    for s in samples:
        result = extractor.extract(s, max_entities=5)
        assert isinstance(result, list)
        assert len(result) >= 1, f"No entities extracted for {s.image_path}"
        assert all(isinstance(e, str) and e for e in result)


@_live_vlm
def test_live_qwen_answer_spatial_question() -> None:
    """Verifies the model produces a non-empty answer for the spatial question."""
    samples = DemoDataset(_SAMPLES_JSON).load()
    sample = samples[0]

    model = QwenVLModel(model_name="Qwen/Qwen2.5-VL-7B-Instruct")
    answer = model.answer(image_path=sample.image_path, question=sample.question)

    assert isinstance(answer, str)
    assert len(answer) > 0


@_live_vlm
def test_live_qwen_output_is_deterministic() -> None:
    """do_sample=False (greedy decoding) should give identical output on two runs."""
    samples = DemoDataset(_SAMPLES_JSON).load()
    sample = samples[0]

    model = QwenVLModel(model_name="Qwen/Qwen2.5-VL-7B-Instruct", do_sample=False)
    extractor = EntityExtractor(model)

    result1 = extractor.extract(sample, max_entities=5)
    result2 = extractor.extract(sample, max_entities=5)

    assert result1 == result2
