from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


def _best_answer_from_record(record: dict[str, Any]) -> str | None:
    # Common OK-VQA / VQA-style fields.
    candidates = [
        record.get("answer"),
        record.get("answers"),
        record.get("multiple_choice_answer"),
    ]

    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()

        if isinstance(value, list) and value:
            normalized_answers: list[str] = []

            for item in value:
                if isinstance(item, str) and item.strip():
                    normalized_answers.append(item.strip())
                elif isinstance(item, dict):
                    answer = item.get("answer")
                    if isinstance(answer, str) and answer.strip():
                        normalized_answers.append(answer.strip())

            if normalized_answers:
                return Counter(normalized_answers).most_common(1)[0][0]

    return None


def _image_path_from_record(
    *,
    record: dict[str, Any],
    image_dir: str | Path,
    sample_id: str,
    image_extension: str = ".jpg",
) -> str:
    image_dir_path = Path(image_dir)
    image_dir_path.mkdir(parents=True, exist_ok=True)

    image_value = (
        record.get("image")
        or record.get("image_path")
        or record.get("image_id")
        or record.get("imageId")
    )

    if image_value is None:
        raise ValueError("OK-VQA record is missing image field.")

    # Hugging Face datasets often return a PIL image object.
    if isinstance(image_value, Image.Image):
        output_path = image_dir_path / f"{sample_id}{image_extension}"
        image_value.convert("RGB").save(output_path)
        return str(output_path)

    # Some datasets expose image objects with a filename.
    filename = getattr(image_value, "filename", None)
    if isinstance(filename, str) and filename:
        source_path = Path(filename)
        if source_path.exists():
            return str(source_path)

    image_text = str(image_value)

    if image_text.lower().endswith((".jpg", ".jpeg", ".png")):
        maybe_path = Path(image_text)
        if maybe_path.exists():
            return str(maybe_path)
        return str(image_dir_path / maybe_path.name)

    return str(image_dir_path / f"{image_text}{image_extension}")


def convert_okvqa_records_to_samples(
    *,
    records: list[dict[str, Any]],
    image_dir: str | Path,
    max_samples: int,
    image_extension: str = ".jpg",
    keyword_filter: list[str] | None = None,
    exclude_keywords: list[str] | None = None,
) -> list[dict[str, str]]:
    include_terms = [keyword.lower() for keyword in keyword_filter or []]
    exclude_terms = [keyword.lower() for keyword in exclude_keywords or []]

    samples: list[dict[str, str]] = []

    for idx, record in enumerate(records):
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            continue

        question_lower = question.lower()

        if include_terms and not any(term in question_lower for term in include_terms):
            continue

        if exclude_terms and any(term in question_lower for term in exclude_terms):
            continue

        gold_answer = _best_answer_from_record(record)
        if not gold_answer:
            continue

        question_id = (
            record.get("question_id")
            or record.get("questionId")
            or record.get("id")
            or idx
        )

        sample_id = f"okvqa_{question_id}"

        try:
            image_path = _image_path_from_record(
                record=record,
                image_dir=image_dir,
                sample_id=sample_id,
                image_extension=image_extension,
            )
        except ValueError:
            continue

        samples.append(
            {
                "sample_id": sample_id,
                "image_path": image_path,
                "question": question.strip(),
                "gold_answer": gold_answer,
            }
        )

        if len(samples) >= max_samples:
            break

    return samples


def write_samples_json(samples: list[dict[str, str]], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(samples, indent=2), encoding="utf-8")