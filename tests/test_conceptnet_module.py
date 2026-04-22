from __future__ import annotations

from vlm_kg_physical_reasoning.retrieval.conceptnet_client import (
    ConceptNetClientError,
    ConceptNetEdge,
)
from vlm_kg_physical_reasoning.retrieval.conceptnet_module import (
    ConceptNetEntityModule,
    ConceptNetModuleResult,
)
from vlm_kg_physical_reasoning.retrieval.node_mapper import NodeMapper


class _StubClient:
    """Minimal in-memory client implementing ConceptNetClientProtocol."""

    def __init__(self, edges_by_uri: dict[str, list[ConceptNetEdge]]) -> None:
        self.edges_by_uri = edges_by_uri
        self.calls: list[tuple[str, int | None]] = []

    def fetch_edges(
        self, node_uri: str, limit: int | None = None
    ) -> list[ConceptNetEdge]:
        self.calls.append((node_uri, limit))
        return list(self.edges_by_uri.get(node_uri, []))


class _ErrorClient:
    def __init__(self, message: str = "boom") -> None:
        self.message = message
        self.calls: list[str] = []

    def fetch_edges(
        self, node_uri: str, limit: int | None = None
    ) -> list[ConceptNetEdge]:
        self.calls.append(node_uri)
        raise ConceptNetClientError(self.message)


def _edge(subject: str, relation: str, obj: str, weight: float = 1.0) -> ConceptNetEdge:
    return ConceptNetEdge(
        subject=subject, relation=relation, object=obj, weight=weight, uri=None
    )


def test_module_returns_edges_for_list_of_entities() -> None:
    client = _StubClient(
        {
            "/c/en/cup": [_edge("cup", "atlocation", "table", 2.0)],
            "/c/en/table": [_edge("table", "usedfor", "holding things", 1.5)],
        }
    )
    module = ConceptNetEntityModule(node_mapper=NodeMapper(), client=client)

    edges = module.fetch_edges_for_entities(["cup", "table"])

    assert len(edges) == 2
    assert {(e.subject, e.relation, e.object) for e in edges} == {
        ("cup", "atlocation", "table"),
        ("table", "usedfor", "holding things"),
    }


def test_module_dedupes_edges_across_entities() -> None:
    shared = _edge("cup", "atlocation", "table", 2.0)
    client = _StubClient({"/c/en/cup": [shared], "/c/en/mug": [shared]})
    module = ConceptNetEntityModule(client=client)

    edges = module.fetch_edges_for_entities(["cup", "mug"])

    assert len(edges) == 1


def test_module_falls_back_to_candidate_uris_when_primary_empty() -> None:
    client = _StubClient(
        {"/c/en/table": [_edge("table", "usedfor", "holding things", 1.0)]}
    )
    module = ConceptNetEntityModule(client=client)

    result = module.query(["wooden table"])

    assert len(result.edges) == 1
    primary_uris = [c[0] for c in client.calls]
    assert primary_uris[0] == "/c/en/wooden_table"
    assert "/c/en/table" in primary_uris
    assert result.reports[0].used_uri == "/c/en/table"


def test_module_can_disable_candidate_fallback() -> None:
    client = _StubClient(
        {"/c/en/table": [_edge("table", "usedfor", "holding things", 1.0)]}
    )
    module = ConceptNetEntityModule(client=client, try_candidate_uris=False)

    result = module.query(["wooden table"])

    assert result.edges == []
    assert client.calls == [("/c/en/wooden_table", 20)]


def test_module_collects_per_entity_errors_without_raising() -> None:
    module = ConceptNetEntityModule(client=_ErrorClient(message="net down"))

    result = module.query(["cup"])

    assert result.edges == []
    assert result.has_errors is True
    assert "cup" in result.errors[0]
    assert result.reports[0].error == "net down"


def test_module_reports_unmappable_entities() -> None:
    module = ConceptNetEntityModule(client=_StubClient({}))

    result = module.query(["!!!", "cup"])

    assert any(r.error == "entity_not_mappable" for r in result.reports)
    assert any(r.entity == "cup" for r in result.reports)


def test_module_query_returns_rich_result_type() -> None:
    client = _StubClient({"/c/en/cup": [_edge("cup", "atlocation", "table", 2.0)]})
    module = ConceptNetEntityModule(client=client)

    result = module.query(["cup"])

    assert isinstance(result, ConceptNetModuleResult)
    assert result.has_errors is False
    assert result.reports[0].mapped_node is not None
    assert result.reports[0].mapped_node.primary_uri == "/c/en/cup"
    assert result.reports[0].edge_count == 1


def test_module_uses_default_dependencies_when_none_provided() -> None:
    module = ConceptNetEntityModule()

    assert module.node_mapper is not None
    assert module.client is not None
    assert module.max_edges_per_node == 20
