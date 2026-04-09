from __future__ import annotations

from urllib.parse import urljoin

from pydantic import BaseModel
import requests


class ConceptNetEdge(BaseModel):
    subject: str
    relation: str
    object: str
    weight: float
    uri: str | None = None


class ConceptNetClientError(RuntimeError):
    """Raised when a ConceptNet request fails."""


class ConceptNetClient:
    """Minimal client for fetching and normalizing ConceptNet edges."""

    def __init__(
        self,
        base_url: str = "https://api.conceptnet.io",
        timeout_seconds: float = 10.0,
        language: str = "en",
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_seconds = timeout_seconds
        self.language = language
        self.session = session or requests.Session()

    def fetch_edges(self, node_uri: str, limit: int | None = None) -> list[ConceptNetEdge]:
        query_url = urljoin(self.base_url, "query")
        params = {"node": node_uri}
        if limit is not None:
            params["limit"] = limit

        try:
            response = self.session.get(query_url, params=params, timeout=self.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise ConceptNetClientError("Failed to fetch ConceptNet edges.") from exc

        normalized_edges: list[ConceptNetEdge] = []
        for raw_edge in payload.get("edges", []):
            edge = self._normalize_edge(raw_edge)
            if edge is not None:
                normalized_edges.append(edge)

        return normalized_edges

    def _normalize_edge(self, raw_edge: dict[str, object]) -> ConceptNetEdge | None:
        start = raw_edge.get("start")
        end = raw_edge.get("end")
        rel = raw_edge.get("rel")

        if not isinstance(start, dict) or not isinstance(end, dict) or not isinstance(rel, dict):
            return None

        start_id = str(start.get("@id", ""))
        end_id = str(end.get("@id", ""))
        if not start_id.startswith(f"/c/{self.language}/") or not end_id.startswith(f"/c/{self.language}/"):
            return None

        subject = self._extract_label(start, start_id)
        relation = self._extract_label(rel, str(rel.get("@id", "")))
        obj = self._extract_label(end, end_id)

        return ConceptNetEdge(
            subject=subject,
            relation=relation,
            object=obj,
            weight=float(raw_edge.get("weight", 0.0)),
            uri=str(raw_edge.get("@id", "")) or None,
        )

    @staticmethod
    def _extract_label(payload: dict[str, object], fallback_uri: str) -> str:
        label = payload.get("label")
        if isinstance(label, str) and label.strip():
            return label.strip().lower()

        return fallback_uri.rstrip("/").split("/")[-1].replace("_", " ").lower()
