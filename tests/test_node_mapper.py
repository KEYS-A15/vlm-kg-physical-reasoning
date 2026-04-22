from __future__ import annotations

from vlm_kg_physical_reasoning.retrieval.node_mapper import MappedNode, NodeMapper


def test_node_mapper_normalizes_simple_entity() -> None:
    mapper = NodeMapper()

    nodes = mapper.map(["Cup"])

    assert len(nodes) == 1
    assert isinstance(nodes[0], MappedNode)
    assert nodes[0].entity == "Cup"
    assert nodes[0].text == "cup"
    assert nodes[0].primary_uri == "/c/en/cup"
    assert nodes[0].candidate_uris == ("/c/en/cup",)
    assert nodes[0].language == "en"


def test_node_mapper_strips_punctuation_and_collapses_spaces() -> None:
    mapper = NodeMapper()

    [node] = mapper.map(["  Wooden, Table!! "])

    assert node.text == "wooden_table"
    assert node.primary_uri == "/c/en/wooden_table"


def test_node_mapper_emits_fallback_candidates_for_multiword_entity() -> None:
    mapper = NodeMapper()

    [node] = mapper.map(["a wooden table"])

    assert node.candidate_uris[0] == "/c/en/a_wooden_table"
    assert "/c/en/wooden_table" in node.candidate_uris
    assert node.candidate_uris[-1] == "/c/en/table"


def test_node_mapper_returns_empty_list_for_blank_input() -> None:
    mapper = NodeMapper()

    assert mapper.map([""]) == []
    assert mapper.map(["   "]) == []
    assert mapper.map(["!!!"]) == []


def test_node_mapper_dedupes_repeated_entities() -> None:
    mapper = NodeMapper()

    nodes = mapper.map(["cup", "Cup", "  cup  "])

    assert len(nodes) == 1
    assert nodes[0].primary_uri == "/c/en/cup"


def test_node_mapper_backward_compat_returns_primary_uris() -> None:
    mapper = NodeMapper()

    assert mapper.map_entities(["cup", "table"]) == ["/c/en/cup", "/c/en/table"]


def test_node_mapper_respects_language_setting() -> None:
    mapper = NodeMapper(language="es")

    [node] = mapper.map(["taza"])

    assert node.primary_uri == "/c/es/taza"
    assert node.language == "es"
