from __future__ import annotations

import json
from pathlib import Path

from vlm_kg_physical_reasoning.data.gqa_adapter import convert_gqa_to_samples


def test_convert_gqa_to_samples(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.json"
    image_dir = tmp_path / "images"

    questions = {
        "123": {
            "imageId": "img_001",
            "question": "Where is the cup?",
            "answer": "on the table",
        },
        "456": {
            "imageId": "img_002",
            "question": "What color is the car?",
            "answer": "red",
        },
    }

    questions_path.write_text(json.dumps(questions), encoding="utf-8")

    samples = convert_gqa_to_samples(
        questions_path=questions_path,
        image_dir=image_dir,
        max_samples=10,
        keyword_filter=["where"],
    )

    assert len(samples) == 1
    assert samples[0]["sample_id"] == "gqa_123"
    assert samples[0]["image_path"].endswith("img_001.jpg")
    assert samples[0]["question"] == "Where is the cup?"
    assert samples[0]["gold_answer"] == "on the table"