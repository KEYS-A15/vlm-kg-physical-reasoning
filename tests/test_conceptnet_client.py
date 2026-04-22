from __future__ import annotations

import pytest
import requests

from vlm_kg_physical_reasoning.retrieval.conceptnet_client import ConceptNetClient, ConceptNetClientError


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class _FailingSession:
    def get(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise requests.RequestException("boom")


class _SuccessfulSession:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def get(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return _Response(self.payload)


def test_conceptnet_client_normalizes_edges() -> None:
    client = ConceptNetClient(
        session=_SuccessfulSession(
            {
                "edges": [
                    {
                        "@id": "/a/[/r/AtLocation/,/c/en/cup/,/c/en/table/]",
                        "start": {"@id": "/c/en/cup", "label": "cup"},
                        "rel": {"@id": "/r/AtLocation", "label": "AtLocation"},
                        "end": {"@id": "/c/en/table", "label": "table"},
                        "weight": 2.5,
                    },
                    {
                        "@id": "/a/[/r/AtLocation/,/c/es/taza/,/c/en/table/]",
                        "start": {"@id": "/c/es/taza", "label": "taza"},
                        "rel": {"@id": "/r/AtLocation", "label": "AtLocation"},
                        "end": {"@id": "/c/en/table", "label": "table"},
                        "weight": 1.0,
                    },
                ]
            }
        )
    )

    edges = client.fetch_edges("/c/en/cup", limit=5)

    assert len(edges) == 1
    assert edges[0].subject == "cup"
    assert edges[0].relation == "atlocation"
    assert edges[0].object == "table"
    assert edges[0].weight == 2.5


def test_conceptnet_client_raises_clean_error_on_request_failure() -> None:
    client = ConceptNetClient(
        session=_FailingSession(), max_retries=0, backoff_seconds=0.0
    )

    with pytest.raises(ConceptNetClientError):
        client.fetch_edges("/c/en/cup", limit=5)


class _FlakySession:
    def __init__(self, payload: dict, fail_first: int) -> None:
        self.payload = payload
        self.fail_first = fail_first
        self.calls = 0

    def get(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.calls += 1
        if self.calls <= self.fail_first:
            raise requests.RequestException("transient")
        return _Response(self.payload)


class _CountingSession:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def get(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.calls += 1
        return _Response(self.payload)


def _single_edge_payload() -> dict:
    return {
        "edges": [
            {
                "@id": "/a/[/r/AtLocation/,/c/en/cup/,/c/en/table/]",
                "start": {"@id": "/c/en/cup", "label": "cup"},
                "rel": {"@id": "/r/AtLocation", "label": "AtLocation"},
                "end": {"@id": "/c/en/table", "label": "table"},
                "weight": 2.5,
            }
        ]
    }


def test_conceptnet_client_retries_transient_failures() -> None:
    session = _FlakySession(_single_edge_payload(), fail_first=2)
    client = ConceptNetClient(
        session=session, max_retries=2, backoff_seconds=0.0
    )

    edges = client.fetch_edges("/c/en/cup", limit=5)

    assert len(edges) == 1
    assert session.calls == 3


def test_conceptnet_client_caches_repeated_requests() -> None:
    session = _CountingSession(_single_edge_payload())
    client = ConceptNetClient(session=session, max_retries=0)

    first = client.fetch_edges("/c/en/cup", limit=5)
    second = client.fetch_edges("/c/en/cup", limit=5)

    assert first == second
    assert session.calls == 1


def test_conceptnet_client_filters_by_min_edge_weight() -> None:
    payload = {
        "edges": [
            {
                "@id": "/a/heavy",
                "start": {"@id": "/c/en/cup", "label": "cup"},
                "rel": {"@id": "/r/AtLocation", "label": "AtLocation"},
                "end": {"@id": "/c/en/table", "label": "table"},
                "weight": 3.0,
            },
            {
                "@id": "/a/light",
                "start": {"@id": "/c/en/cup", "label": "cup"},
                "rel": {"@id": "/r/RelatedTo", "label": "RelatedTo"},
                "end": {"@id": "/c/en/mug", "label": "mug"},
                "weight": 0.5,
            },
        ]
    }
    client = ConceptNetClient(
        session=_SuccessfulSession(payload),
        max_retries=0,
        min_edge_weight=1.0,
    )

    edges = client.fetch_edges("/c/en/cup")

    assert [e.relation for e in edges] == ["atlocation"]


def test_conceptnet_client_skips_malformed_edges() -> None:
    payload = {
        "edges": [
            "not a dict",
            {"start": {"@id": "/c/en/cup"}, "rel": "missing", "end": {}},
            {
                "@id": "/a/ok",
                "start": {"@id": "/c/en/cup", "label": "cup"},
                "rel": {"@id": "/r/AtLocation", "label": "AtLocation"},
                "end": {"@id": "/c/en/table", "label": "table"},
                "weight": 1.0,
            },
        ]
    }
    client = ConceptNetClient(
        session=_SuccessfulSession(payload), max_retries=0
    )

    edges = client.fetch_edges("/c/en/cup")

    assert len(edges) == 1
    assert edges[0].subject == "cup"
