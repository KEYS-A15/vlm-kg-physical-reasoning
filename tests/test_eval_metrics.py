from __future__ import annotations
from typing import Any
from vlm_kg_physical_reasoning.eval.metrics import (
    contains_gold,
    exact_match,
    normalize_answer,
    score_system,
    token_overlap_f1,
)


def test_normalize_answer() -> None:
    assert normalize_answer("Left of the books!") == "left of the books"


def test_exact_match() -> None:
    assert exact_match("left of the books", "Left of the books!") is True
    assert exact_match("the cup is left of the books", "left of the books") is False


def test_contains_gold() -> None:
    assert contains_gold("The cup is left of the books.", "left of the books") is True
    assert contains_gold("The cup is on the desk.", "left of the books") is False


def test_token_overlap_f1() -> None:
    score = token_overlap_f1("The cup is left of the books.", "left of the books")
    assert score > 0.0


def test_score_system() -> None:
    rows: list[dict[str, Any]] = [
        {"final_answer": "left of the books", "gold_answer": "left of the books"},
        {"final_answer": "The cup is left of the books.", "gold_answer": "left of the books"},
        {"final_answer": "on the desk", "gold_answer": "left of the books"},
    ]

    scores = score_system("baseline", rows)

    assert scores.total == 3
    assert scores.exact_match_count == 1
    assert scores.contains_gold_count == 2
    assert scores.avg_token_f1 > 0.0