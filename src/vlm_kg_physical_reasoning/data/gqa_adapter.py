from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_gqa_questions(path: str | Path) -> dict[str, dict[str, Any]]:
    file_path = Path(path)
    payload = json.loads(file_path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError(f"GQA questions file must be a JSON object: {file_path}")

    return {
        str(question_id): question_payload
        for question_id, question_payload in payload.items()
        if isinstance(question_payload, dict)
    }


def gqa_image_path(
    *,
    image_dir: str | Path,
    image_id: str,
    extension: str = ".jpg",
) -> str:
    return str(Path(image_dir) / f"{image_id}{extension}")


def convert_gqa_record_to_sample(
    *,
    question_id: str,
    record: dict[str, Any],
    image_dir: str | Path,
    image_extension: str = ".jpg",
) -> dict[str, str]:
    image_id = record.get("imageId")
    question = record.get("question")
    answer = record.get("answer")

    if not isinstance(image_id, str) or not image_id:
        raise ValueError(f"Missing imageId for GQA question_id={question_id}")

    if not isinstance(question, str) or not question:
        raise ValueError(f"Missing question for GQA question_id={question_id}")

    if not isinstance(answer, str) or not answer:
        raise ValueError(f"Missing answer for GQA question_id={question_id}")

    return {
        "sample_id": f"gqa_{question_id}",
        "image_path": gqa_image_path(
            image_dir=image_dir,
            image_id=image_id,
            extension=image_extension,
        ),
        "question": question,
        "gold_answer": answer,
    }


def convert_gqa_to_samples(
    *,
    questions_path: str | Path,
    image_dir: str | Path,
    max_samples: int,
    image_extension: str = ".jpg",
    keyword_filter: list[str] | None = None,
    exclude_keywords: list[str] | None = None,
) -> list[dict[str, str]]:
    questions = load_gqa_questions(questions_path)

    samples: list[dict[str, str]] = []
    include_terms = [keyword.lower() for keyword in keyword_filter or []]
    exclude_terms = [keyword.lower() for keyword in exclude_keywords or []]

    for question_id, record in questions.items():
        question = record.get("question")
        if not isinstance(question, str):
            continue

        question_lower = question.lower()

        if include_terms and not any(keyword in question_lower for keyword in include_terms):
            continue

        if exclude_terms and any(keyword in question_lower for keyword in exclude_terms):
            continue

        try:
            sample = convert_gqa_record_to_sample(
                question_id=question_id,
                record=record,
                image_dir=image_dir,
                image_extension=image_extension,
            )
        except ValueError:
            continue

        samples.append(sample)

        if len(samples) >= max_samples:
            break

    return samples