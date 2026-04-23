from __future__ import annotations

from pathlib import Path

import typer

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


def main() -> None:
    app()


def _resolve_sample_file(config: AppConfig, sample_file: Path | None) -> Path:
    if sample_file is not None:
        return sample_file

    if config.paths.sample_data:
        return Path(config.paths.sample_data)

    raise typer.BadParameter("Provide --sample-file or set paths.sample_data in the config.")


def _load_sample(config: AppConfig, sample_file: Path | None, sample_id: str | None) -> Sample:
    dataset = DemoDataset(_resolve_sample_file(config, sample_file))
    samples = dataset.load()

    if sample_id is None:
        return samples[0]

    for sample in samples:
        if sample.sample_id == sample_id:
            return sample

    raise typer.BadParameter(f"Sample id '{sample_id}' was not found in the sample file.")


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
    config_path: Path = typer.Option("configs/default.yaml", "--config", help="Path to the YAML config."),
    sample_file: Path | None = typer.Option(None, "--sample-file", help="Path to a sample JSON file."),
    sample_id: str | None = typer.Option(None, "--sample-id", help="Optional sample id to select from a file."),
    model_name: str | None = typer.Option(None, "--model-name", help="Optional model override."),
) -> None:
    """Run the direct VLM baseline on one sample."""

    config = load_config(config_path)
    sample = _load_sample(config=config, sample_file=sample_file, sample_id=sample_id)
    pipeline = BaselinePipeline(vlm=_build_vlm(config, model_name))
    result = pipeline.run(sample)

    output_dir = ensure_dir(config.paths.prediction_output_dir)
    output_path = output_dir / f"{sample.sample_id}_baseline.json"
    write_json(output_path, result)

    typer.echo(f"Sample: {sample.sample_id}")
    typer.echo(f"Answer: {result['final_answer']}")
    typer.echo(f"Saved prediction: {output_path}")


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
            "question_type": trace.question_type,
            "final_answer": trace.final_answer,
            "trace_path": str(trace_path),
        },
    )
    write_json(trace_path, trace.model_dump())

    typer.echo(f"Sample: {sample.sample_id}")
    typer.echo(f"Answer: {trace.final_answer}")
    typer.echo(f"Saved prediction: {prediction_path}")
    typer.echo(f"Saved trace: {trace_path}")
