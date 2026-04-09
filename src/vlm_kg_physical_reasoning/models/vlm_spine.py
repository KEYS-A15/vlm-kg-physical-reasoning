from __future__ import annotations

from typing import Protocol


class VLMBackbone(Protocol):
    """Shared interface for the baseline and KG pipelines."""

    def answer(
        self,
        image_path: str,
        question: str,
        evidence: list[str] | None = None,
    ) -> str:
        """Generate a plain-text answer for an image-question pair."""

    def extract_entities(
        self,
        image_path: str,
        question: str,
        max_entities: int,
    ) -> list[str]:
        """Extract a short list of concrete entities from the image-question pair."""
