from __future__ import annotations

from pathlib import Path

import typer
from typing import Any
from rich.console import Console
from rich.table import Table

from vlm_kg_physical_reasoning.config import AppConfig, load_config
from vlm_kg_physical_reasoning.data.dataset import DemoDataset
from vlm_kg_physical_reasoning.data.sample import Sample
from vlm_kg_physical_reasoning.extraction.entity_extraction import EntityExtractor
from vlm_kg_physical_reasoning.models.qwen_vl import QwenVLModel
from vlm_kg_physical_reasoning.pipelines.baseline import BaselinePipeline
from vlm_kg_physical_reasoning.pipelines.kg_naive import NaiveKGPipeline
from vlm_kg_physical_reasoning.retrieval.basic_retriever import BasicRetriever
from vlm_kg_physical_reasoning.retrieval.conceptnet_client import make_conceptnet_client
from vlm_kg_physical_reasoning.retrieval.node_mapper import NodeMapper
from vlm_kg_physical_reasoning.retrieval.question_classifier import QuestionClassifier
from vlm_kg_physical_reasoning.tracing.trace_builder import TraceBuilder
from vlm_kg_physical_reasoning.utils.io import ensure_dir, write_json


app = typer.Typer(no_args_is_help=True, help="RKG-VLM phase 2 commands.")
console = Console()

def main() -> None:
    app()


def _resolve_sample_file(config: AppConfig, sample_file: Path | None) -> Path:
    if sample_file is not None:
        return sample_file

    if config.paths.sample_data:
        return Path(config.paths.sample_data)

    raise typer.BadParameter("Provide --sample-file or set paths.sample_data in the config.")

def _load_samples(
    config: AppConfig,
    sample_file: Path | None = None,
) -> list[Sample]:
    dataset_path = sample_file or config.paths.sample_data
    dataset = DemoDataset(dataset_path)
    return dataset.load()

def _load_sample(
    config: AppConfig,
    sample_file: Path | None = None,
    sample_id: str | None = None,
) -> Sample:
    samples = _load_samples(config=config, sample_file=sample_file)

    if sample_id is None:
        return samples[0]

    for sample in samples:
        if sample.sample_id == sample_id:
            return sample

    raise typer.BadParameter(f"No sample found with sample_id={sample_id}")


def _build_vlm(config: AppConfig, model_name_override: str | None) -> QwenVLModel:
    model_name = model_name_override or config.model.name
    return QwenVLModel(
        model_name=model_name,
        generation_max_new_tokens=config.model.generation_max_new_tokens,
        entity_extraction_max_new_tokens=config.model.entity_extraction_max_new_tokens,
        do_sample=config.model.do_sample,
        temperature=config.model.temperature,
        device_map=config.model.device_map,
        torch_dtype=config.model.torch_dtype,
    )


@app.command("run-baseline")
def run_baseline(
    config_path: str = typer.Option("configs/default.yaml", "--config"),
    sample_file: Path | None = typer.Option(None, "--sample-file"),
    sample_id: str | None = typer.Option(None, "--sample-id"),
    model_name: str | None = typer.Option(None, "--model-name"),
) -> None:
    config = load_config(config_path)
    samples = _load_samples(config=config, sample_file=sample_file)

    if sample_id is not None:
        samples = [sample for sample in samples if sample.sample_id == sample_id]
        if not samples:
            raise typer.BadParameter(f"No sample found with sample_id={sample_id}")

    pipeline = BaselinePipeline(vlm=_build_vlm(config, model_name))
    output_dir = ensure_dir(config.paths.prediction_output_dir)

    table = Table(title="Baseline VLM Results", show_lines=True)
    table.add_column("Sample ID", style="cyan", no_wrap=True)
    table.add_column("Question", style="white")
    table.add_column("Gold", style="yellow")
    table.add_column("Prediction", style="green")
    table.add_column("Output File", style="magenta")

    for sample in samples:
        console.print(f"[blue]Running baseline for:[/blue] {sample.sample_id}")
        result = pipeline.run(sample)
        output_path = output_dir / f"{sample.sample_id}_baseline.json"

        payload: dict[str, Any] = result.model_dump()
        write_json(output_path, payload)

        table.add_row(
            result.sample_id,
            result.question,
            result.gold_answer or "-",
            result.final_answer,
            str(output_path),
        )

    console.print(table)
    console.print(
        f"[bold green]Completed baseline run for {len(samples)} sample(s).[/bold green]"
    )


def _edge_to_dict(edge: Any) -> dict[str, Any]:
    if hasattr(edge, "model_dump"):
        dumped = edge.model_dump()
        if isinstance(dumped, dict):
            return dumped

    if isinstance(edge, dict):
        return edge

    return {
        "subject": getattr(edge, "subject", ""),
        "relation": getattr(edge, "relation", ""),
        "object": getattr(edge, "object", ""),
        "weight": getattr(edge, "weight", 0.0),
        "uri": getattr(edge, "uri", None),
    }


def _edge_to_text(edge: Any) -> str:
    edge_dict = _edge_to_dict(edge)

    subject = str(edge_dict.get("subject", "")).strip()
    relation = str(edge_dict.get("relation", "")).strip()
    obj = str(edge_dict.get("object", "")).strip()

    if not subject and not relation and not obj:
        return "-"

    return f"{subject} {relation} {obj}".strip()


@app.command("run-kg-naive")
def run_kg_naive(
    config_path: Path = typer.Option(
        "configs/default.yaml",
        "--config",
        help="Path to the YAML config.",
    ),
    sample_file: Path | None = typer.Option(
        None,
        "--sample-file",
        help="Path to a sample JSON file.",
    ),
    sample_id: str | None = typer.Option(
        None,
        "--sample-id",
        help="Optional sample id to select from a file.",
    ),
    model_name: str | None = typer.Option(
        None,
        "--model-name",
        help="Optional model override.",
    ),
) -> None:
    """Run the naive ConceptNet-augmented VLM pipeline."""

    config = load_config(config_path)
    samples = _load_samples(config=config, sample_file=sample_file)

    if sample_id is not None:
        samples = [sample for sample in samples if sample.sample_id == sample_id]
        if not samples:
            raise typer.BadParameter(f"No sample found with sample_id={sample_id}")

    vlm = _build_vlm(config, model_name)
    conceptnet_client = make_conceptnet_client(
        base_url=config.retrieval.conceptnet.base_url,
        timeout_seconds=config.retrieval.conceptnet.timeout_seconds,
        language=config.retrieval.conceptnet.language,
        gradio_space_url=config.retrieval.conceptnet.gradio_space_url,
    )
    retriever = BasicRetriever(
        client=conceptnet_client,
        max_edges_per_node=config.retrieval.conceptnet.max_edges_per_node,
        overlap_weight=config.retrieval.overlap_weight,
    )

    pipeline = NaiveKGPipeline(
        vlm=vlm,
        entity_extractor=EntityExtractor(vlm),
        node_mapper=NodeMapper(),
        retriever=retriever,
        question_classifier=QuestionClassifier(),
        trace_builder=TraceBuilder(),
        max_entities=config.pipeline.max_entities,
        max_evidence_triples=config.pipeline.max_evidence_triples,
    )

    prediction_dir = ensure_dir(config.paths.prediction_output_dir)
    trace_dir = ensure_dir(config.paths.trace_output_dir)

    table = Table(title="KG-Naive Pipeline Results", show_lines=True)
    table.add_column("Sample ID", style="cyan", no_wrap=True)
    table.add_column("Type", style="blue", no_wrap=True)
    table.add_column("Entities", style="white")
    table.add_column("Selected Evidence", style="magenta")
    table.add_column("Gold", style="yellow")
    table.add_column("Prediction", style="green")
    table.add_column("Trace", style="dim")

    for sample in samples:
        console.print(f"[blue]Running KG-naive for:[/blue] {sample.sample_id}")

        trace = pipeline.run(sample)

        prediction_path = prediction_dir / f"{sample.sample_id}_kg_naive.json"
        trace_path = trace_dir / f"{sample.sample_id}_kg_naive_trace.json"

        prediction_payload: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "gold_answer": sample.gold_answer,
            "question_type": trace.question_type,
            "entities": trace.entities,
            "selected_evidence": [_edge_to_dict(edge) for edge in trace.selected_evidence],
            "final_answer": trace.final_answer,
            "trace_path": str(trace_path),
        }

        write_json(prediction_path, prediction_payload)
        write_json(trace_path, trace.model_dump())

        evidence_preview = "\n".join(
            _edge_to_text(edge) for edge in trace.selected_evidence[:3]
        )
        if not evidence_preview:
            evidence_preview = "-"

        table.add_row(
            sample.sample_id,
            trace.question_type,
            ", ".join(trace.entities) if trace.entities else "-",
            evidence_preview,
            sample.gold_answer or "-",
            trace.final_answer,
            str(trace_path),
        )

    console.print(table)
    console.print(
        f"[bold green]Completed KG-naive run for {len(samples)} sample(s).[/bold green]"
    )

