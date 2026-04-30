from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.table import Table

from vlm_kg_physical_reasoning.data.okvqa_adapter import (
    convert_okvqa_records_to_samples,
    write_samples_json,
)


console = Console()


OKVQA_KG_INCLUDE = [
    "why",
    "what is used",
    "used for",
    "use for",
    "made of",
    "material",
    "what kind",
    "what type",
    "which kind",
    "which type",
    "what can",
    "what do",
    "purpose",
    "function",
    "called",
]

OKVQA_KG_EXCLUDE = [
    "left",
    "right",
    "under",
    "above",
    "below",
    "behind",
    "front",
    "where",
    "side",
    "color",
    "colour",
    "how many",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an OK-VQA subset in RKG-VLM sample format."
    )
    parser.add_argument(
        "--dataset",
        default="lmms-lab/OK-VQA",
        help="Hugging Face dataset id.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split to load. Try validation, val, test, or train depending on HF config.",
    )
    parser.add_argument(
        "--image-dir",
        default="data/okvqa/images",
        help="Fallback image directory if records contain image IDs instead of paths.",
    )
    parser.add_argument("--out", default="data/okvqa/okvqa_subset.json")
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument(
        "--no-keyword-filter",
        action="store_true",
        help="Disable KG-oriented question filtering.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset = load_dataset(args.dataset, split=args.split)
    records = [dict(row) for row in dataset]

    samples = convert_okvqa_records_to_samples(
        records=records,
        image_dir=args.image_dir,
        max_samples=args.max_samples,
        keyword_filter=None if args.no_keyword_filter else OKVQA_KG_INCLUDE,
        exclude_keywords=None if args.no_keyword_filter else OKVQA_KG_EXCLUDE,
    )

    write_samples_json(samples, args.out)

    table = Table(title="Prepared OK-VQA Subset", show_lines=True)
    table.add_column("Dataset")
    table.add_column("Split")
    table.add_column("N", justify="right")
    table.add_column("Output")
    table.add_column("First Sample")

    first_sample = samples[0]["sample_id"] if samples else "-"
    table.add_row(args.dataset, args.split, str(len(samples)), str(Path(args.out)), first_sample)

    console.print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())