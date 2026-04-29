from __future__ import annotations

import re
from dataclasses import dataclass


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def normalize_answer(value: str | None) -> str:
    if value is None:
        return ""

    tokens = _TOKEN_PATTERN.findall(value.lower())
    return " ".join(tokens)


def exact_match(prediction: str | None, gold: str | None) -> bool:
    pred = normalize_answer(prediction)
    target = normalize_answer(gold)

    if not target:
        return False

    return pred == target


def contains_gold(prediction: str | None, gold: str | None) -> bool:
    pred = normalize_answer(prediction)
    target = normalize_answer(gold)

    if not target:
        return False

    return target in pred


def token_overlap_f1(prediction: str | None, gold: str | None) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)
    overlap = len(pred_set & gold_set)

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_set)
    recall = overlap / len(gold_set)

    return (2 * precision * recall) / (precision + recall)


@dataclass(frozen=True)
class SystemScores:
    system_name: str
    total: int
    exact_match_count: int
    contains_gold_count: int
    avg_token_f1: float

    @property
    def exact_match_accuracy(self) -> float:
        return self.exact_match_count / self.total if self.total else 0.0

    @property
    def contains_gold_accuracy(self) -> float:
        return self.contains_gold_count / self.total if self.total else 0.0


def score_system(system_name: str, rows: list[dict[str, object]]) -> SystemScores:
    exact_count = 0
    contains_count = 0
    f1_values: list[float] = []

    for row in rows:
        prediction = row.get("final_answer")
        gold = row.get("gold_answer")

        pred_text = prediction if isinstance(prediction, str) else None
        gold_text = gold if isinstance(gold, str) else None

        if exact_match(pred_text, gold_text):
            exact_count += 1

        if contains_gold(pred_text, gold_text):
            contains_count += 1

        f1_values.append(token_overlap_f1(pred_text, gold_text))

    avg_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0

    return SystemScores(
        system_name=system_name,
        total=len(rows),
        exact_match_count=exact_count,
        contains_gold_count=contains_count,
        avg_token_f1=avg_f1,
    )