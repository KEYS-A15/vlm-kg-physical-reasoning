"""High-level ConceptNet + node-mapping module.

This module is the public entry point for the "entity → ConceptNet edges"
contribution. It composes a :class:`NodeMapper` and any
:class:`ConceptNetClientProtocol` implementation and exposes a single
function-style API:

    list[str]  ->  list[ConceptNetEdge]

It is designed to slot into a larger pipeline without coupling to the rest
of the codebase. Other contributors can:

    * Plug in a different mapper by subclassing / replacing ``NodeMapper``.
    * Plug in a different client (cache, mock, alternative KG) as long as
      it satisfies ``ConceptNetClientProtocol``.
    * Read the rich :class:`ConceptNetModuleResult` for tracing / debugging
      without changing the simple ``list[str] -> list[ConceptNetEdge]`` flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vlm_kg_physical_reasoning.retrieval.conceptnet_client import (
    ConceptNetClient,
    ConceptNetClientError,
    ConceptNetClientProtocol,
    ConceptNetEdge,
)
from vlm_kg_physical_reasoning.retrieval.node_mapper import MappedNode, NodeMapper


@dataclass(slots=True)
class EntityFetchReport:
    """Per-entity bookkeeping describing how an entity was resolved."""

    entity: str
    mapped_node: MappedNode | None
    used_uri: str | None
    edge_count: int
    error: str | None = None


@dataclass(slots=True)
class ConceptNetModuleResult:
    """Rich result returned by :meth:`ConceptNetEntityModule.query`."""

    edges: list[ConceptNetEdge]
    reports: list[EntityFetchReport] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)


class ConceptNetEntityModule:
    """Map entity strings to ConceptNet edges with graceful degradation.

    The module never raises on a per-entity failure; failures are recorded
    in the returned :class:`ConceptNetModuleResult`. This keeps it safe to
    drop into pipelines where the VLM should still be able to answer even
    when ConceptNet is partially unreachable.

    Args:
        node_mapper: Maps raw entity strings to ConceptNet URIs. Defaults
            to a fresh :class:`NodeMapper`.
        client: Any object implementing :class:`ConceptNetClientProtocol`.
            Defaults to :class:`ConceptNetClient` with library defaults.
        max_edges_per_node: Upper bound passed to ``client.fetch_edges``
            for each candidate URI.
        try_candidate_uris: When ``True`` (default), retry with each
            candidate URI from the mapper if the primary URI yields no
            edges. Useful for multi-word entities like ``"wooden table"``.
    """

    def __init__(
        self,
        node_mapper: NodeMapper | None = None,
        client: ConceptNetClientProtocol | None = None,
        max_edges_per_node: int = 20,
        try_candidate_uris: bool = True,
    ) -> None:
        self.node_mapper = node_mapper or NodeMapper()
        self.client = client or ConceptNetClient()
        self.max_edges_per_node = max_edges_per_node
        self.try_candidate_uris = try_candidate_uris

    def fetch_edges_for_entities(self, entities: list[str]) -> list[ConceptNetEdge]:
        """Spec-mandated API: ``list[str] -> list[ConceptNetEdge]``."""

        return self.query(entities).edges

    def query(self, entities: list[str]) -> ConceptNetModuleResult:
        """Like :meth:`fetch_edges_for_entities` but returns rich diagnostics."""

        mapped_nodes = self.node_mapper.map(entities)
        mapped_by_entity: dict[str, MappedNode] = {
            node.entity: node for node in mapped_nodes
        }

        reports: list[EntityFetchReport] = []
        errors: list[str] = []
        all_edges: list[ConceptNetEdge] = []

        for entity in entities:
            mapped = mapped_by_entity.get(entity)
            if mapped is None or not mapped.is_valid:
                reports.append(
                    EntityFetchReport(
                        entity=entity,
                        mapped_node=None,
                        used_uri=None,
                        edge_count=0,
                        error="entity_not_mappable",
                    )
                )
                continue

            edges, used_uri, error = self._fetch_for_node(mapped)
            if error is not None:
                errors.append(f"{entity}: {error}")

            all_edges.extend(edges)
            reports.append(
                EntityFetchReport(
                    entity=entity,
                    mapped_node=mapped,
                    used_uri=used_uri,
                    edge_count=len(edges),
                    error=error,
                )
            )

        return ConceptNetModuleResult(
            edges=self._dedupe_edges(all_edges),
            reports=reports,
            errors=errors,
        )

    def _fetch_for_node(
        self, mapped: MappedNode
    ) -> tuple[list[ConceptNetEdge], str | None, str | None]:
        uris_to_try: list[str]
        if self.try_candidate_uris:
            uris_to_try = list(mapped.candidate_uris)
        else:
            uris_to_try = [mapped.primary_uri]

        last_error: str | None = None
        for uri in uris_to_try:
            try:
                edges = self.client.fetch_edges(uri, limit=self.max_edges_per_node)
            except ConceptNetClientError as exc:
                last_error = str(exc)
                continue

            if edges:
                return edges, uri, None

        return [], uris_to_try[0] if uris_to_try else None, last_error

    @staticmethod
    def _dedupe_edges(edges: list[ConceptNetEdge]) -> list[ConceptNetEdge]:
        seen: set[tuple[str, str, str]] = set()
        unique: list[ConceptNetEdge] = []

        for edge in edges:
            key = (edge.subject, edge.relation, edge.object)
            if key in seen:
                continue
            seen.add(key)
            unique.append(edge)

        return unique


__all__ = [
    "ConceptNetEntityModule",
    "ConceptNetModuleResult",
    "EntityFetchReport",
]
