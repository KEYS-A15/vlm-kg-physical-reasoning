from __future__ import annotations

import re
from dataclasses import dataclass

_NON_WORD_PATTERN = re.compile(r"[^a-z0-9\s]+")
_SPACE_PATTERN = re.compile(r"\s+")

_DEFAULT_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "and",
        "or",
        "with",
        "for",
        "to",
        "in",
        "on",
        "at",
        "by",
        "from",
    }
)


@dataclass(slots=True, frozen=True)
class MappedNode:
    """A normalized ConceptNet node mapping for a single entity string.

    Attributes:
        entity: The original entity string as provided by the caller.
        text: The normalized lower-cased text used to build the URI.
        primary_uri: The first ConceptNet URI to try (e.g. ``/c/en/wooden_table``).
        candidate_uris: Ordered list of fallback URIs to try if the primary
            URI returns no edges. Includes ``primary_uri`` as the first entry.
        language: ConceptNet language code (e.g. ``en``).
    """

    entity: str
    text: str
    primary_uri: str
    candidate_uris: tuple[str, ...]
    language: str

    @property
    def is_valid(self) -> bool:
        return bool(self.text) and bool(self.primary_uri)


class NodeMapper:
    """Map raw entity strings to ConceptNet node URIs.

    The mapper is intentionally side-effect free and returns multiple URI
    candidates per entity so downstream retrievers can fall back when the
    primary URI has no edges (e.g. ``/c/en/wooden_table`` has no edges but
    ``/c/en/table`` does).
    """

    def __init__(
        self,
        language: str = "en",
        stopwords: frozenset[str] | None = None,
    ) -> None:
        self.language = language
        self.stopwords = stopwords if stopwords is not None else _DEFAULT_STOPWORDS

    def map(self, entities: list[str]) -> list[MappedNode]:
        """Map each entity to a ``MappedNode``. Invalid entities are skipped."""

        mapped: list[MappedNode] = []
        seen_primary: set[str] = set()

        for entity in entities:
            node = self._map_one(entity)
            if node is None or node.primary_uri in seen_primary:
                continue
            seen_primary.add(node.primary_uri)
            mapped.append(node)

        return mapped

    def map_entities(self, entities: list[str]) -> list[str]:
        """Backward-compatible API: return only primary URIs."""

        return [node.primary_uri for node in self.map(entities)]

    def _map_one(self, entity: str) -> MappedNode | None:
        normalized = self.normalize_entity(entity)
        if not normalized:
            return None

        candidates = self._build_candidates(normalized)
        if not candidates:
            return None

        primary = self._to_uri(candidates[0])
        candidate_uris = tuple(self._to_uri(c) for c in candidates)

        return MappedNode(
            entity=entity,
            text=candidates[0],
            primary_uri=primary,
            candidate_uris=candidate_uris,
            language=self.language,
        )

    def _build_candidates(self, normalized_text: str) -> list[str]:
        """Generate candidate URI slugs in priority order without duplicates."""

        tokens = [t for t in normalized_text.split("_") if t]
        if not tokens:
            return []

        candidates: list[str] = []
        seen: set[str] = set()

        def _add(value: str) -> None:
            if value and value not in seen:
                seen.add(value)
                candidates.append(value)

        _add("_".join(tokens))

        content_tokens = [t for t in tokens if t not in self.stopwords]
        if content_tokens and content_tokens != tokens:
            _add("_".join(content_tokens))

        if len(content_tokens) > 1:
            _add(content_tokens[-1])
        elif len(tokens) > 1:
            _add(tokens[-1])

        return candidates

    def _to_uri(self, slug: str) -> str:
        return f"/c/{self.language}/{slug}"

    @staticmethod
    def normalize_entity(entity: str) -> str:
        cleaned = _NON_WORD_PATTERN.sub(" ", entity.strip().lower())
        collapsed = _SPACE_PATTERN.sub(" ", cleaned).strip()
        return collapsed.replace(" ", "_")


__all__ = ["MappedNode", "NodeMapper"]
