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
    client = ConceptNetClient(session=_FailingSession())

    with pytest.raises(ConceptNetClientError):
        client.fetch_edges("/c/en/cup", limit=5)
