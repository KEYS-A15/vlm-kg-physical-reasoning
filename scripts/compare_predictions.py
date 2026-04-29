from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from vlm_kg_physical_reasoning.eval.comparison import (
    build_comparison_rows,
    load_system_predictions,
)
from vlm_kg_physical_reasoning.eval.metrics import score_system
from vlm_kg_physical_reasoning.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline, KG-naive, and question-aware KG predictions."
    )
    parser.add_argument(
        "--prediction-dir",
        default="outputs/predictions",
        help="Directory containing saved prediction JSON files.",
    )
    parser.add_argument(
        "--out",
        default="outputs/eval/comparison_summary.json",
        help="Path to save comparison summary JSON.",
    )
    return parser.parse_args()


def _evidence_preview(value: object, max_items: int = 3) -> str:
    if not isinstance(value, list) or not value:
        return "-"

    lines: list[str] = []
    for item in value[:max_items]:
        if isinstance(item, dict):
            subject = str(item.get("subject", "")).strip()
            relation = str(item.get("relation", "")).strip()
            obj = str(item.get("object", "")).strip()
            text = f"{subject} {relation} {obj}".strip()
            if text:
                lines.append(text)

    return "\n".join(lines) if lines else "-"


def _scores_to_dict(score: Any) -> dict[str, object]:
    return {
        "system_name": score.system_name,
        "total": score.total,
        "exact_match_count": score.exact_match_count,
        "exact_match_accuracy": score.exact_match_accuracy,
        "contains_gold_count": score.contains_gold_count,
        "contains_gold_accuracy": score.contains_gold_accuracy,
        "avg_token_f1": score.avg_token_f1,
    }


def main() -> int:
    configure_logging("INFO")
    args = parse_args()
    prediction_dir = Path(args.prediction_dir)

    baseline_rows = load_system_predictions(prediction_dir, "baseline")
    naive_rows = load_system_predictions(prediction_dir, "kg_naive")
    qa_rows = load_system_predictions(prediction_dir, "kg_question_aware")

    scores = [
        score_system("baseline", baseline_rows),
        score_system("kg_naive", naive_rows),
        score_system("kg_question_aware", qa_rows),
    ]

    comparison_rows = build_comparison_rows(
        baseline_rows=baseline_rows,
        naive_rows=naive_rows,
        question_aware_rows=qa_rows,
    )

    score_table = Table(title="System-Level Evaluation", show_lines=True)
    score_table.add_column("System", style="cyan")
    score_table.add_column("N", justify="right")
    score_table.add_column("Exact", justify="right")
    score_table.add_column("Contains Gold", justify="right")
    score_table.add_column("Avg Token F1", justify="right")

    for score in scores:
        score_table.add_row(
            score.system_name,
            str(score.total),
            f"{score.exact_match_accuracy:.3f}",
            f"{score.contains_gold_accuracy:.3f}",
            f"{score.avg_token_f1:.3f}",
        )

    console.print(score_table)

    comparison_table = Table(title="Prediction Comparison", show_lines=True)
    comparison_table.add_column("Sample", style="cyan", no_wrap=True)
    comparison_table.add_column("Gold", style="yellow")
    comparison_table.add_column("Baseline", style="white")
    comparison_table.add_column("KG-Naive", style="magenta")
    comparison_table.add_column("Question-Aware", style="green")
    comparison_table.add_column("Naive Evidence", style="purple")
    comparison_table.add_column("QA Evidence", style="purple")

    for row in comparison_rows:
        comparison_table.add_row(
            str(row.get("sample_id", "")),
            str(row.get("gold_answer", "")),
            str(row.get("baseline_answer", "")),
            str(row.get("kg_naive_answer", "")),
            str(row.get("kg_question_aware_answer", "")),
            _evidence_preview(row.get("naive_evidence")),
            _evidence_preview(row.get("question_aware_evidence")),
        )

    console.print(comparison_table)

    output = {
        "scores": [_scores_to_dict(score) for score in scores],
        "comparison_rows": comparison_rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    logger.info("Saved comparison summary: %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())