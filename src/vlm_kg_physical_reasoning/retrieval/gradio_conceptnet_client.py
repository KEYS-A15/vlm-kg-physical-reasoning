"""ConceptNet edge retrieval via a Hugging Face Gradio Space (normalized DB).

The official public API (``https://api.conceptnet.io``) is a separate service from
e.g. `cstr/conceptnet_normalized <https://huggingface.co/spaces/cstr/conceptnet_normalized>`__.
Those Spaces expose a **Gradio** API, not the JSON-LD ``/query?node=`` endpoint,
so this client calls the Space's ``/run_query`` function and maps rows to
``ConceptNetEdge``.

Typical space URL: ``https://cstr-conceptnet-normalized.hf.space`` (from the
Space subdomain ``cstr-conceptnet-normalized``).
"""

from __future__ import annotations
import contextlib
import io
import re
from typing import Any

from vlm_kg_physical_reasoning.retrieval.conceptnet_client import (
    ConceptNetClientError,
    ConceptNetClientProtocol,
    ConceptNetEdge,
)

_C_PATH = re.compile(r"(/c/[a-z]{2,3}/[^/?#]+)")


def _conceptnet_path(uri: str) -> str | None:
    m = _C_PATH.search(uri)
    if not m:
        return None
    p = m.group(1).rstrip("/")
    return p if p.startswith("/c/") else None


class GradioConceptNetClient(ConceptNetClientProtocol):
    """Fetches edges from a Gradio ``run_query``-style API on a Hugging Face Space."""

    def __init__(
        self,
        space_url: str = "https://cstr-conceptnet-normalized.hf.space",
        default_language: str = "en",
        cache_enabled: bool = True,
    ) -> None:
        from gradio_client import Client  # type: ignore[import-not-found]

        self._space_url = space_url.rstrip("/")
        self.default_language = default_language
        self.cache_enabled = cache_enabled
        self._cache: dict[tuple[str, int | None], list[ConceptNetEdge]] = {}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self._client: Any = Client(self._space_url)
        except Exception as exc:  # noqa: BLE001
            raise ConceptNetClientError(
                f"Failed to connect to Gradio space {self._space_url}: {exc}"
            ) from exc

    def clear_cache(self) -> None:
        self._cache.clear()

    def fetch_edges(
        self, node_uri: str, limit: int | None = None
    ) -> list[ConceptNetEdge]:
        want = 20 if limit is None else int(limit)
        if want < 1:
            want = 1
        if want > 500:
            want = 500
        key = (node_uri, want)
        if self.cache_enabled and key in self._cache:
            return list(self._cache[key])

        path = _conceptnet_path(node_uri)
        if not path:
            return []

        segs = [p for p in path.split("/") if p]
        if len(segs) < 3 or segs[0] != "c":
            return []
        lang, term = segs[1], segs[2]

        # ``run_query`` uses LIKE on ``http://conceptnet.io/c/{lang}/term%`` — request
        # extra rows so filtering to the exact start node still leaves enough.
        request_limit = min(500, max(want * 8, want))

        try:
            result = self._client.predict(
                term,
                lang,
                "",  # any relation
                "",
                lang,
                float(request_limit),
                api_name="/run_query",
            )
        except Exception as exc:  # noqa: BLE001
            raise ConceptNetClientError(
                f"Gradio run_query failed for {node_uri}: {exc}"
            ) from exc

        if not isinstance(result, (list, tuple)) or len(result) < 1:
            return []

        first = result[0] if result else None
        rows: list[Any] = []
        headers: list[str] = []
        if isinstance(first, dict):
            drows = first.get("data")
            if isinstance(drows, list):
                rows = drows
            h = first.get("headers")
            if isinstance(h, list):
                headers = [str(x) for x in h]

        if not rows or not headers:
            edges: list[ConceptNetEdge] = []
            if self.cache_enabled:
                self._cache[key] = list(edges)
            return edges

        want_path = path
        out: list[ConceptNetEdge] = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) != len(headers):
                continue
            d = dict(zip(headers, row))
            start_url = d.get("start_url", "")
            if _conceptnet_path(str(start_url)) != want_path:
                continue
            subj = str(d.get("start_label", "")).strip().lower()
            rel = str(d.get("relation", "")).strip().lower()
            obj = str(d.get("end_label", "")).strip().lower()
            wraw = d.get("weight", 0.0)
            try:
                w = float(wraw)
            except (TypeError, ValueError):
                w = 0.0
            rur = d.get("relation_url")
            rurl = str(rur).strip() if rur is not None else None
            out.append(
                ConceptNetEdge(
                    subject=subj,
                    relation=rel,
                    object=obj,
                    weight=w,
                    uri=rurl or None,
                )
            )
            if len(out) >= want:
                break

        if self.cache_enabled:
            self._cache[key] = list(out)
        return out
