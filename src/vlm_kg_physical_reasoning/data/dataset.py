from __future__ import annotations

import json
from pathlib import Path

from vlm_kg_physical_reasoning.data.sample import Sample


class DemoDataset:
    """Load one sample or a small list of samples from JSON."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[Sample]:
        with self.path.open("r", encoding="utf-8") as file:
            raw = json.load(file)

        if isinstance(raw, dict) and "samples" in raw:
            samples = raw["samples"]
        elif isinstance(raw, dict):
            samples = [raw]
        elif isinstance(raw, list):
            samples = raw
        else:
            raise ValueError("Sample file must contain a sample object, a list of samples, or {'samples': [...]} .")

        return [Sample.model_validate(item) for item in samples]
