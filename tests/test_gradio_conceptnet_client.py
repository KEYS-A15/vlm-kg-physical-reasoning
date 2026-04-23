from __future__ import annotations

from unittest.mock import MagicMock, patch

from vlm_kg_physical_reasoning.retrieval.gradio_conceptnet_client import (
    GradioConceptNetClient,
)


def test_gradio_client_maps_dataframe_to_edges() -> None:
    mock_predict = MagicMock(
        return_value=(
            {
                "headers": [
                    "start_label",
                    "relation",
                    "end_label",
                    "weight",
                    "start_url",
                    "end_url",
                    "relation_url",
                ],
                "data": [
                    [
                        "cup",
                        "AtLocation",
                        "table",
                        4.0,
                        "http://conceptnet.io/c/en/cup",
                        "http://conceptnet.io/c/en/table",
                        "http://conceptnet.io/r/AtLocation",
                    ],
                ],
            },
            "ok",
        )
    )
    mock_client_instance = MagicMock()
    mock_client_instance.predict = mock_predict

    with patch(
        "gradio_client.Client",
        return_value=mock_client_instance,
    ):
        client = GradioConceptNetClient(space_url="https://example.hf.space")

    edges = client.fetch_edges("/c/en/cup", limit=10)

    assert len(edges) == 1
    assert edges[0].subject == "cup"
    assert edges[0].relation == "atlocation"
    assert edges[0].object == "table"
    assert edges[0].weight == 4.0


def test_make_conceptnet_client_uses_gradio_when_url_set() -> None:
    from vlm_kg_physical_reasoning.retrieval.conceptnet_client import (
        make_conceptnet_client,
    )
    from vlm_kg_physical_reasoning.retrieval.gradio_conceptnet_client import (
        GradioConceptNetClient,
    )

    with patch(
        "gradio_client.Client",
        return_value=MagicMock(predict=MagicMock(return_value=({"headers": [], "data": []}, ""))),
    ):
        c = make_conceptnet_client(gradio_space_url="https://x.hf.space")
    assert isinstance(c, GradioConceptNetClient)
