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


@app.command("run-kg-naive")
def run_kg_naive(
    config_path: Path = typer.Option("configs/default.yaml", "--config", help="Path to the YAML config."),
    sample_file: Path | None = typer.Option(None, "--sample-file", help="Path to a sample JSON file."),
    sample_id: str | None = typer.Option(None, "--sample-id", help="Optional sample id to select from a file."),
    model_name: str | None = typer.Option(None, "--model-name", help="Optional model override."),
) -> None:
    """Run the naive ConceptNet-augmented VLM pipeline on one sample."""

    config = load_config(config_path)
    sample = _load_sample(config=config, sample_file=sample_file, sample_id=sample_id)
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
        overlap_weight=config.retrieval.overlap_weight
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

    trace = pipeline.run(sample)

    prediction_dir = ensure_dir(config.paths.prediction_output_dir)
    trace_dir = ensure_dir(config.paths.trace_output_dir)
    prediction_path = prediction_dir / f"{sample.sample_id}_kg_naive.json"
    trace_path = trace_dir / f"{sample.sample_id}_kg_naive_trace.json"

    write_json(
        prediction_path,
        {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "gold_answer": sample.gold_answer,
            "question_type": trace.question_type,
            "final_answer": trace.final_answer,
            "trace_path": str(trace_path),
        },
    )
    write_json(trace_path, trace.model_dump())

    typer.echo(f"Sample: {sample.sample_id}")
    typer.echo(f"Answer: {trace.final_answer}")
    typer.echo(f"Gold answer: {sample.gold_answer}")
    typer.echo(f"Saved prediction: {prediction_path}")
    typer.echo(f"Saved trace: {trace_path}")
