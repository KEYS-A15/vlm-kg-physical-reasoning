"""Live demo of the ConceptNet entity module.

Usage (PowerShell):
    uv run python scripts/demo_conceptnet_module.py
    uv run python scripts/demo_conceptnet_module.py cup table "wooden table" knife
    uv run python scripts/demo_conceptnet_module.py --official-api

By default this uses the Hugging Face Gradio mirror (see ``configs/default.yaml``
``gradio_space_url``), not the public REST host. Pass ``--official-api`` to call
``https://api.conceptnet.io`` only (may 502 if their gateway is down).

The script fetches remote data and prints:
    1. How each entity string was normalized and mapped to URIs.
    2. Which URI actually returned edges (after candidate fallback).
    3. A table of the strongest fetched edges.
    4. Any errors that occurred (network failures, etc.).

Requires network access. The default entities are a small physical-reasoning
set so the output stays readable.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from vlm_kg_physical_reasoning.config import load_config
from vlm_kg_physical_reasoning.retrieval.conceptnet_client import make_conceptnet_client
from vlm_kg_physical_reasoning.retrieval.conceptnet_module import ConceptNetEntityModule
from vlm_kg_physical_reasoning.retrieval.node_mapper import NodeMapper

DEFAULT_ENTITIES = ["cup", "table", "wooden table", "knife", "glass"]
TOP_EDGES_TO_SHOW = 10


def main() -> int:
    console = Console()
    args = [a for a in sys.argv[1:] if a != ""]
    use_official = "--official-api" in args
    args = [a for a in args if a != "--official-api"]
    entities = args or DEFAULT_ENTITIES

    app_config = load_config("configs/default.yaml")
    space_url = "" if use_official else app_config.retrieval.conceptnet.gradio_space_url

    mode = "official REST (api.conceptnet.io)" if use_official else f"Gradio mirror ({space_url or 'n/a'})"
    console.print(
        Panel.fit(
            f"[bold]Backend:[/bold] {mode}\n[bold]Entities:[/bold] {entities}",
            title="ConceptNet entity module demo",
            border_style="cyan",
        )
    )

    mapper = NodeMapper()
    console.print("\n[bold]1. Node mapping[/bold]")
    mapping_table = Table(show_lines=False)
    mapping_table.add_column("Entity", style="bold")
    mapping_table.add_column("Normalized text")
    mapping_table.add_column("Candidate URIs (in priority order)")
    for node in mapper.map(entities):
        mapping_table.add_row(
            node.entity, node.text, "\n".join(node.candidate_uris)
        )
    console.print(mapping_table)

    console.print("\n[bold]2. Fetching edges from ConceptNet ...[/bold]")
    client = make_conceptnet_client(
        base_url=app_config.retrieval.conceptnet.base_url,
        timeout_seconds=15.0,
        max_retries=2,
        backoff_seconds=0.5,
        cache_enabled=True,
        language=app_config.retrieval.conceptnet.language,
        gradio_space_url=space_url,
    )
    module = ConceptNetEntityModule(
        node_mapper=mapper, client=client, max_edges_per_node=20
    )
    result = module.query(entities)

    console.print("\n[bold]3. Per-entity report[/bold]")
    report_table = Table(show_lines=False)
    report_table.add_column("Entity", style="bold")
    report_table.add_column("URI used")
    report_table.add_column("# edges", justify="right")
    report_table.add_column("Error", style="red")
    for r in result.reports:
        report_table.add_row(
            r.entity,
            r.used_uri or "-",
            str(r.edge_count),
            r.error or "",
        )
    console.print(report_table)

    console.print(
        f"\n[bold]4. Top {TOP_EDGES_TO_SHOW} edges by weight "
        f"(out of {len(result.edges)} unique)[/bold]"
    )
    edges_table = Table(show_lines=False)
    edges_table.add_column("Subject", style="cyan")
    edges_table.add_column("Relation", style="magenta")
    edges_table.add_column("Object", style="green")
    edges_table.add_column("Weight", justify="right")
    for edge in sorted(result.edges, key=lambda e: e.weight, reverse=True)[
        :TOP_EDGES_TO_SHOW
    ]:
        edges_table.add_row(
            edge.subject, edge.relation, edge.object, f"{edge.weight:.2f}"
        )
    console.print(edges_table)

    if result.has_errors:
        console.print("\n[bold red]Errors:[/bold red]")
        for err in result.errors:
            console.print(f"  - {err}")
    else:
        console.print("\n[green]No errors.[/green]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
