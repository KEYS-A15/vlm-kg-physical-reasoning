from __future__ import annotations
from typing import Any
from vlm_kg_physical_reasoning.eval.comparison import build_comparison_rows


def test_build_comparison_rows_aligns_by_sample_id() -> None:
    baseline: list[dict[str, Any]] = [
        {
            "sample_id": "demo_001",
            "question": "Where is the cup?",
            "gold_answer": "left of the books",
            "final_answer": "in front of the books",
        }
    ]

    naive: list[dict[str, Any]] = [
        {
            "sample_id": "demo_001",
            "question": "Where is the cup?",
            "gold_answer": "left of the books",
            "final_answer": "on the desk",
            "selected_evidence": [
                {"subject": "books", "relation": "atlocation", "object": "desk"}
            ],
        }
    ]

    qa = [
        {
            "sample_id": "demo_001",
            "question": "Where is the cup?",
            "gold_answer": "left of the books",
            "final_answer": "in front of the books",
            "selected_evidence": [],
        }
    ]

    rows = build_comparison_rows(
        baseline_rows=baseline,
        naive_rows=naive,
        question_aware_rows=qa,
    )

    assert len(rows) == 1
    assert rows[0]["sample_id"] == "demo_001"
    assert rows[0]["baseline_answer"] == "in front of the books"
    assert rows[0]["kg_naive_answer"] == "on the desk"
    assert rows[0]["kg_question_aware_answer"] == "in front of the books"
    assert rows[0]["naive_evidence"] == [
        {"subject": "books", "relation": "atlocation", "object": "desk"}
    ]