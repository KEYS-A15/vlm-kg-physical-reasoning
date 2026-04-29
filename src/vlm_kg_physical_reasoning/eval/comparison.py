from __future__ import annotations

import json
from pathlib import Path


SYSTEM_SUFFIXES: dict[str, str] = {
    "baseline": "_baseline.json",
    "kg_naive": "_kg_naive.json",
    "kg_question_aware": "_kg_question_aware.json",
}


def load_json_file(path: str | Path) -> dict[str, object]:
    file_path = Path(path)

    with file_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {file_path}")

    return payload


def load_system_predictions(
    prediction_dir: str | Path,
    system_name: str,
) -> list[dict[str, object]]:
    directory = Path(prediction_dir)

    if system_name not in SYSTEM_SUFFIXES:
        raise ValueError(f"Unknown system name: {system_name}")

    suffix = SYSTEM_SUFFIXES[system_name]
    rows: list[dict[str, object]] = []

    if not directory.exists():
        return rows

    for path in sorted(directory.glob(f"*{suffix}")):
        rows.append(load_json_file(path))

    return rows


def index_by_sample_id(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}

    for row in rows:
        sample_id = row.get("sample_id")
        if isinstance(sample_id, str):
            indexed[sample_id] = row

    return indexed


def build_comparison_rows(
    baseline_rows: list[dict[str, object]],
    naive_rows: list[dict[str, object]],
    question_aware_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    baseline_by_id = index_by_sample_id(baseline_rows)
    naive_by_id = index_by_sample_id(naive_rows)
    qa_by_id = index_by_sample_id(question_aware_rows)

    sample_ids = sorted(set(baseline_by_id) | set(naive_by_id) | set(qa_by_id))
    comparison_rows: list[dict[str, object]] = []

    for sample_id in sample_ids:
        baseline = baseline_by_id.get(sample_id, {})
        naive = naive_by_id.get(sample_id, {})
        qa = qa_by_id.get(sample_id, {})

        gold = baseline.get("gold_answer") or naive.get("gold_answer") or qa.get("gold_answer")
        question = baseline.get("question") or naive.get("question") or qa.get("question") or ""

        comparison_rows.append(
            {
                "sample_id": sample_id,
                "question": question,
                "gold_answer": gold,
                "baseline_answer": baseline.get("final_answer", ""),
                "kg_naive_answer": naive.get("final_answer", ""),
                "kg_question_aware_answer": qa.get("final_answer", ""),
                "naive_question_type": naive.get("question_type", ""),
                "question_aware_question_type": qa.get("question_type", ""),
                "naive_evidence": naive.get("selected_evidence", []),
                "question_aware_evidence": qa.get("selected_evidence", []),
                "baseline_trace_path": baseline.get("trace_path", ""),
                "kg_naive_trace_path": naive.get("trace_path", ""),
                "kg_question_aware_trace_path": qa.get("trace_path", ""),
            }
        )

    return comparison_rows