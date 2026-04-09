from __future__ import annotations

import json

from vlm_kg_physical_reasoning.config import load_config
from vlm_kg_physical_reasoning.data.dataset import DemoDataset


def test_load_default_config() -> None:
    config = load_config("configs/default.yaml")

    assert config.project.name == "RKG-VLM"
    assert config.model.name == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert config.pipeline.max_evidence_triples == 5
    assert config.retrieval.conceptnet.max_edges_per_node == 10


def test_dataset_loads_single_object_and_list(tmp_path) -> None:
    single_path = tmp_path / "single.json"
    list_path = tmp_path / "list.json"
    single_payload = {
        "sample_id": "single",
        "image_path": "image.png",
        "question": "What is on the table?",
    }
    list_payload = [
        single_payload,
        {
            "sample_id": "second",
            "image_path": "image-2.png",
            "question": "Why would the book fall?",
        },
    ]

    single_path.write_text(json.dumps(single_payload), encoding="utf-8")
    list_path.write_text(json.dumps(list_payload), encoding="utf-8")

    single_samples = DemoDataset(single_path).load()
    list_samples = DemoDataset(list_path).load()

    assert len(single_samples) == 1
    assert single_samples[0].sample_id == "single"
    assert len(list_samples) == 2
    assert list_samples[1].sample_id == "second"
