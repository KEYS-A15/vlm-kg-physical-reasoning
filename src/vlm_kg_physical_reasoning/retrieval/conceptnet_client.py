from __future__ import annotations

import time
from typing import Protocol, runtime_checkable
from urllib.parse import urljoin

import requests
from pydantic import BaseModel


class ConceptNetEdge(BaseModel):
    """Normalized representation of a single ConceptNet assertion."""

    subject: str
    relation: str
    object: str
    weight: float
    uri: str | None = None


class ConceptNetClientError(RuntimeError):
    """Raised when a ConceptNet request fails after exhausting retries."""


@runtime_checkable
class ConceptNetClientProtocol(Protocol):
    """Minimal contract any ConceptNet-like client must satisfy.

    Implementations may wrap the public HTTP API, a local cache, a static
    fixture for tests, or a different KG entirely - so long as they return
    edges in the normalized ``ConceptNetEdge`` schema.
    """

    def fetch_edges(
        self, node_uri: str, limit: int | None = None
    ) -> list[ConceptNetEdge]: ...


class ConceptNetClient:
    """HTTP client for fetching and normalizing ConceptNet edges.

    Features:
        * Retries with exponential backoff on transient HTTP errors.
        * Optional in-memory cache keyed by ``(node_uri, limit)``.
        * Language filtering on both endpoints of every edge.
        * Optional minimum-weight filter applied during normalization.
    """

    def __init__(
        self,
        base_url: str = "https://api.conceptnet.io",
        timeout_seconds: float = 10.0,
        language: str = "en",
        session: requests.Session | None = None,
        max_retries: int = 2,
        backoff_seconds: float = 0.5,
        min_edge_weight: float = 0.0,
        cache_enabled: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_seconds = timeout_seconds
        self.language = language
        self.session = session or requests.Session()
        self.max_retries = max(0, max_retries)
        self.backoff_seconds = max(0.0, backoff_seconds)
        self.min_edge_weight = min_edge_weight
        self.cache_enabled = cache_enabled
        self._cache: dict[tuple[str, int | None], list[ConceptNetEdge]] = {}

    def fetch_edges(
        self, node_uri: str, limit: int | None = None
    ) -> list[ConceptNetEdge]:
        cache_key = (node_uri, limit)
        if self.cache_enabled and cache_key in self._cache:
            return list(self._cache[cache_key])

        payload = self._request_with_retries(node_uri, limit)
        normalized = self._normalize_payload(payload)

        if self.cache_enabled:
            self._cache[cache_key] = list(normalized)

        return normalized

    def clear_cache(self) -> None:
        self._cache.clear()

    def _request_with_retries(
        self, node_uri: str, limit: int | None
    ) -> dict[str, object]:
        query_url = urljoin(self.base_url, "query")
        params: dict[str, object] = {"node": node_uri}
        if limit is not None:
            params["limit"] = limit

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(
                    query_url, params=params, timeout=self.timeout_seconds
                )
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
                    continue
                raise ConceptNetClientError(
                    f"Failed to fetch ConceptNet edges for {node_uri}: {exc}"
                ) from exc
            else:
                if not isinstance(payload, dict):
                    raise ConceptNetClientError(
                        f"Unexpected ConceptNet payload type: {type(payload).__name__}"
                    )
                return payload

        # Defensive: the loop above always returns or raises.
        raise ConceptNetClientError(
            f"Failed to fetch ConceptNet edges for {node_uri}: {last_exc}"
        )

    def _normalize_payload(self, payload: dict[str, object]) -> list[ConceptNetEdge]:
        raw_edges = payload.get("edges", [])
        if not isinstance(raw_edges, list):
            return []

        normalized: list[ConceptNetEdge] = []
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, dict):
                continue
            edge = self._normalize_edge(raw_edge)
            if edge is not None and edge.weight >= self.min_edge_weight:
                normalized.append(edge)

        return normalized

    def _normalize_edge(
        self, raw_edge: dict[str, object]
    ) -> ConceptNetEdge | None:
        start = raw_edge.get("start")
        end = raw_edge.get("end")
        rel = raw_edge.get("rel")

        if (
            not isinstance(start, dict)
            or not isinstance(end, dict)
            or not isinstance(rel, dict)
        ):
            return None

        start_id = str(start.get("@id", ""))
        end_id = str(end.get("@id", ""))
        language_prefix = f"/c/{self.language}/"
        if not start_id.startswith(language_prefix) or not end_id.startswith(
            language_prefix
        ):
            return None

        try:
            weight = float(raw_edge.get("weight", 0.0))
        except (TypeError, ValueError):
            weight = 0.0

        subject = self._extract_label(start, start_id)
        relation = self._extract_label(rel, str(rel.get("@id", "")))
        obj = self._extract_label(end, end_id)

        if not subject or not relation or not obj:
            return None

        uri_value = str(raw_edge.get("@id", "")) or None

        return ConceptNetEdge(
            subject=subject,
            relation=relation,
            object=obj,
            weight=weight,
            uri=uri_value,
        )

    @staticmethod
    def _extract_label(payload: dict[str, object], fallback_uri: str) -> str:
        label = payload.get("label")
        if isinstance(label, str) and label.strip():
            return label.strip().lower()

        return fallback_uri.rstrip("/").split("/")[-1].replace("_", " ").lower()


__all__ = [
    "ConceptNetClient",
    "ConceptNetClientError",
    "ConceptNetClientProtocol",
    "ConceptNetEdge",
]
