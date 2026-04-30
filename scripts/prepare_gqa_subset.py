from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from vlm_kg_physical_reasoning.data.gqa_adapter import convert_gqa_to_samples


console = Console()


SPATIAL_KEYWORDS = [
    "where",
    "on",
    "under",
    "above",
    "below",
    "behind",
    "front",
    "left",
    "right",
    "between",
    "near",
    "on top",
    "side",
]


KG_CANDIDATE_INCLUDE = [
    "used for",
    "use for",
    "made of",
    "material",
    "can be used",
    "can hold",
    "can contain",
    "able to",
    "purpose",
    "function",
    "kind of food",
    "type of food",
    "kind of animal",
    "type of animal",
    "kind of furniture",
    "type of furniture",
    "kind of vehicle",
    "type of vehicle",
]


KG_CANDIDATE_EXCLUDE = [
    "left",
    "right",
    "under",
    "above",
    "below",
    "behind",
    "front",
    "between",
    "near",
    "where",
    "side",
    "on top",
    "color",
    "colour",
    "who",
    "wearing",
    "holding",
    "called",
    "name of",
    "which place",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a small GQA question subset into RKG-VLM sample format."
    )
    parser.add_argument("--questions", required=True, help="Path to GQA questions JSON.")
    parser.add_argument("--image-dir", required=True, help="Directory containing GQA images.")
    parser.add_argument("--out", default="data/gqa/gqa_subset.json")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--image-extension", default=".jpg")
    parser.add_argument(
        "--preset",
        choices=["spatial", "kg-candidate", "none"],
        default="spatial",
        help="Subset preset to generate.",
    )
    parser.add_argument(
        "--no-keyword-filter",
        action="store_true",
        help="Disable preset keyword filtering.",
    )
    return parser.parse_args()


def _resolve_filters(
    *,
    preset: str,
    no_keyword_filter: bool,
) -> tuple[list[str] | None, list[str] | None]:
    if no_keyword_filter or preset == "none":
        return None, None

    if preset == "spatial":
        return SPATIAL_KEYWORDS, None

    if preset == "kg-candidate":
        return KG_CANDIDATE_INCLUDE, KG_CANDIDATE_EXCLUDE

    raise ValueError(f"Unsupported preset: {preset}")


def main() -> int:
    args = parse_args()

    keyword_filter, exclude_keywords = _resolve_filters(
        preset=args.preset,
        no_keyword_filter=args.no_keyword_filter,
    )

    samples = convert_gqa_to_samples(
        questions_path=args.questions,
        image_dir=args.image_dir,
        max_samples=args.max_samples,
        image_extension=args.image_extension,
        keyword_filter=keyword_filter,
        exclude_keywords=exclude_keywords,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(samples, indent=2), encoding="utf-8")

    table = Table(title="Prepared GQA Subset", show_lines=True)
    table.add_column("Preset")
    table.add_column("N", justify="right")
    table.add_column("Output")
    table.add_column("First Sample")

    first_sample = samples[0]["sample_id"] if samples else "-"
    table.add_row(args.preset, str(len(samples)), str(out_path), first_sample)

    console.print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())