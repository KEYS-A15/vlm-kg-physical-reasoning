from __future__ import annotations

import re


_NON_WORD_PATTERN = re.compile(r"[^a-z0-9\s]+")
_SPACE_PATTERN = re.compile(r"\s+")


class NodeMapper:
    """Map extracted entity text to simple English ConceptNet nodes."""

    def map_entities(self, entities: list[str]) -> list[str]:
        mapped_nodes: list[str] = []

        for entity in entities:
            normalized = self.normalize_entity(entity)
            if normalized:
                mapped_nodes.append(f"/c/en/{normalized}")

        return mapped_nodes

    @staticmethod
    def normalize_entity(entity: str) -> str:
        cleaned = _NON_WORD_PATTERN.sub(" ", entity.strip().lower())
        collapsed = _SPACE_PATTERN.sub(" ", cleaned).strip()
        return collapsed.replace(" ", "_")
