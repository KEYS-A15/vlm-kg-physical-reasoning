from __future__ import annotations

from pathlib import Path

import typer
from typing import Any
import logging
import os
from rich.console import Console
from rich.panel import Panel
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
from vlm_kg_physical_reasoning.retrieval.question_aware_retriever import (
    QuestionAwareRetriever,
)
from vlm_kg_physical_reasoning.eval.metrics import (
    contains_gold,
    token_overlap_f1,
)

app = typer.Typer(
    no_args_is_help=True,
    help="RKG-VLM phase 2 commands.",
    pretty_exceptions_enable=False,
    rich_markup_mode="rich",
    context_settings={
        "color": False,
    },
)
console = Console()

def main() -> None:
    app()

def _silence_external_logs() -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    noisy_loggers = [
        "",
        "httpx",
        "httpcore",
        "urllib3",
        "gradio_client",
        "huggingface_hub",
        "transformers",
        "sentence_transformers",
    ]

    for logger_name in noisy_loggers:
        noisy_logger = logging.getLogger(logger_name)
        noisy_logger.setLevel(logging.ERROR)
        noisy_logger.handlers.clear()
        noisy_logger.propagate = False

def _print_sample_panel(sample: Sample) -> None:
    console.print(
        Panel.fit(
            f"[bold cyan]{sample.sample_id}[/bold cyan]\n"
            f"[white]{sample.question}[/white]\n"
            f"[yellow]Gold:[/yellow] {sample.gold_answer or '-'}",
            title="Sample",
            border_style="cyan",
        )
    )


def _print_step(label: str, style: str) -> None:
    console.print(f"[{style}]▣ {label}[/{style}]")

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

def _run_kg_pipeline(
    *,
    config_path: Path,
    sample_file: Path | None,
    sample_id: str | None,
    model_name: str | None,
    use_question_aware_retrieval: bool,
) -> None:
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

    if use_question_aware_retrieval:
        retriever = QuestionAwareRetriever(
            client=conceptnet_client,
            max_edges_per_node=config.retrieval.conceptnet.max_edges_per_node,
            overlap_weight=config.retrieval.overlap_weight,
            relation_prior_weight=config.retrieval.relation_prior_weight,
            suppress_spatial_kg=config.retrieval.suppress_spatial_kg,
        )
        output_suffix = "kg_question_aware"
        title = "KG Question-Aware Pipeline Results"
    else:
        retriever = BasicRetriever(
            client=conceptnet_client,
            max_edges_per_node=config.retrieval.conceptnet.max_edges_per_node,
            overlap_weight=config.retrieval.overlap_weight,
        )
        output_suffix = "kg_naive"
        title = "KG-Naive Pipeline Results"

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

    table = Table(title=title, show_lines=True)
    table.add_column("Sample ID", style="cyan", no_wrap=True)
    table.add_column("Type", style="cornflower_blue", no_wrap=True)
    table.add_column("Entities", style="white")
    table.add_column("Selected Evidence", style="magenta")
    table.add_column("Gold", style="yellow")
    table.add_column("Prediction", style="green")
    table.add_column("Trace", style="purple")

    for sample in samples:
        console.print(f"[dodger_blue1]Running {output_suffix} for:[/cornflower_blue] {sample.sample_id}")

        trace = pipeline.run(sample)

        prediction_path = prediction_dir / f"{sample.sample_id}_{output_suffix}.json"
        trace_path = trace_dir / f"{sample.sample_id}_{output_suffix}_trace.json"

        prediction_payload: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "gold_answer": sample.gold_answer,
            "question_type": trace.question_type,
            "entities": trace.entities,
            "selected_evidence": [
                _edge_to_dict(edge) for edge in trace.selected_evidence
            ],
            "final_answer": trace.final_answer,
            "trace_path": str(trace_path),
            "retrieval_mode": output_suffix,
        }

        write_json(prediction_path, prediction_payload)
        write_json(trace_path, trace.model_dump())

        evidence_preview = "\n".join(
            _edge_to_text(edge) for edge in trace.selected_evidence[:3]
        )
        if not evidence_preview:
            evidence_preview = "[purple]No KG evidence used[/purple]"

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
        f"[bold green]Completed {output_suffix} run for {len(samples)} sample(s).[/bold green]"
    )

def _build_conceptnet_client_from_config(config: AppConfig) -> Any:
    return make_conceptnet_client(
        base_url=config.retrieval.conceptnet.base_url,
        timeout_seconds=config.retrieval.conceptnet.timeout_seconds,
        language=config.retrieval.conceptnet.language,
        gradio_space_url=config.retrieval.conceptnet.gradio_space_url,
    )


def _build_kg_pipeline(
    *,
    config: AppConfig,
    vlm: QwenVLModel,
    use_question_aware_retrieval: bool,
) -> NaiveKGPipeline:
    conceptnet_client = _build_conceptnet_client_from_config(config)

    if use_question_aware_retrieval:
        retriever = QuestionAwareRetriever(
            client=conceptnet_client,
            max_edges_per_node=config.retrieval.conceptnet.max_edges_per_node,
            overlap_weight=config.retrieval.overlap_weight,
            relation_prior_weight=config.retrieval.relation_prior_weight,
            suppress_spatial_kg=config.retrieval.suppress_spatial_kg,
        )
    else:
        retriever = BasicRetriever(
            client=conceptnet_client,
            max_edges_per_node=config.retrieval.conceptnet.max_edges_per_node,
            overlap_weight=config.retrieval.overlap_weight,
        )

    return NaiveKGPipeline(
        vlm=vlm,
        entity_extractor=EntityExtractor(vlm),
        node_mapper=NodeMapper(),
        retriever=retriever,
        question_classifier=QuestionClassifier(),
        trace_builder=TraceBuilder(),
        max_entities=config.pipeline.max_entities,
        max_evidence_triples=config.pipeline.max_evidence_triples,
    )


def _format_metric(value: float) -> str:
    return f"{value:.3f}"


def _is_contains_gold(prediction: str | None, gold: str | None) -> str:
    return "yes" if contains_gold(prediction, gold) else "no"

@app.command("run-baseline")
def run_baseline(
    config_path: str = typer.Option("configs/default.yaml", "--config"),
    sample_file: Path | None = typer.Option(None, "--sample-file"),
    sample_id: str | None = typer.Option(None, "--sample-id"),
    model_name: str | None = typer.Option(None, "--model-name"),
) -> None:
    _silence_external_logs()
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
        _print_step(f"Running Baseline VLM for {sample.sample_id}", "bold dodger_blue1")
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
    config_path: Path = typer.Option("configs/default.yaml", "--config"),
    sample_file: Path | None = typer.Option(None, "--sample-file"),
    sample_id: str | None = typer.Option(None, "--sample-id"),
    model_name: str | None = typer.Option(None, "--model-name"),
) -> None:
    """Run the naive ConceptNet-augmented VLM pipeline."""
    _silence_external_logs()
    
    _run_kg_pipeline(
        config_path=config_path,
        sample_file=sample_file,
        sample_id=sample_id,
        model_name=model_name,
        use_question_aware_retrieval=False,
    )

@app.command("run-kg-question-aware")
def run_kg_question_aware(
    config_path: Path = typer.Option("configs/default.yaml", "--config"),
    sample_file: Path | None = typer.Option(None, "--sample-file"),
    sample_id: str | None = typer.Option(None, "--sample-id"),
    model_name: str | None = typer.Option(None, "--model-name"),
) -> None:
    """Run the question-aware ConceptNet-augmented VLM pipeline."""
    _silence_external_logs()
    
    _run_kg_pipeline(
        config_path=config_path,
        sample_file=sample_file,
        sample_id=sample_id,
        model_name=model_name,
        use_question_aware_retrieval=True,
    )

@app.command("run-all")
def run_all(
    config_path: Path = typer.Option("configs/default.yaml", "--config"),
    sample_file: Path | None = typer.Option(None, "--sample-file"),
    sample_id: str | None = typer.Option(None, "--sample-id"),
    model_name: str | None = typer.Option(None, "--model-name"),
) -> None:
    """Run baseline, KG-naive, and KG-question-aware per sample with immediate comparison."""

    _silence_external_logs()
    config = load_config(config_path)
    
    samples = _load_samples(config=config, sample_file=sample_file)

    if sample_id is not None:
        samples = [sample for sample in samples if sample.sample_id == sample_id]
        if not samples:
            raise typer.BadParameter(f"No sample found with sample_id={sample_id}")

    prediction_dir = ensure_dir(config.paths.prediction_output_dir)
    trace_dir = ensure_dir(config.paths.trace_output_dir)

    # Build one VLM and reuse it across all modes.
    vlm = _build_vlm(config, model_name)

    baseline_pipeline = BaselinePipeline(vlm=vlm)
    kg_naive_pipeline = _build_kg_pipeline(
        config=config,
        vlm=vlm,
        use_question_aware_retrieval=False,
    )
    kg_question_aware_pipeline = _build_kg_pipeline(
        config=config,
        vlm=vlm,
        use_question_aware_retrieval=True,
    )

    summary_table = Table(title="Run-All Summary", show_lines=True)
    summary_table.add_column("Sample", style="cyan", no_wrap=True)
    summary_table.add_column("Gold", style="yellow")
    summary_table.add_column("Baseline", style="white")
    summary_table.add_column("KG-Naive", style="magenta")
    summary_table.add_column("Question-Aware", style="green")
    summary_table.add_column("Naive Evidence", style="purple")
    summary_table.add_column("QA Evidence", style="purple")

    for sample in samples:
        _print_sample_panel(sample)

        _print_step("Running Baseline VLM", "bold dodger_blue1")
        baseline_result = baseline_pipeline.run(sample)
        baseline_path = prediction_dir / f"{sample.sample_id}_baseline.json"
        write_json(baseline_path, baseline_result.model_dump())

        _print_step("Running KG-Naive", "bold magenta")
        naive_trace = kg_naive_pipeline.run(sample)
        naive_prediction_path = prediction_dir / f"{sample.sample_id}_kg_naive.json"
        naive_trace_path = trace_dir / f"{sample.sample_id}_kg_naive_trace.json"

        naive_payload: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "gold_answer": sample.gold_answer,
            "question_type": naive_trace.question_type,
            "entities": naive_trace.entities,
            "selected_evidence": [
                _edge_to_dict(edge) for edge in naive_trace.selected_evidence
            ],
            "final_answer": naive_trace.final_answer,
            "trace_path": str(naive_trace_path),
            "retrieval_mode": "kg_naive",
        }
        write_json(naive_prediction_path, naive_payload)
        write_json(naive_trace_path, naive_trace.model_dump())

        _print_step("Running KG Question-Aware", "bold green")
        qa_trace = kg_question_aware_pipeline.run(sample)
        qa_prediction_path = prediction_dir / f"{sample.sample_id}_kg_question_aware.json"
        qa_trace_path = trace_dir / f"{sample.sample_id}_kg_question_aware_trace.json"

        qa_payload: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "gold_answer": sample.gold_answer,
            "question_type": qa_trace.question_type,
            "entities": qa_trace.entities,
            "selected_evidence": [
                _edge_to_dict(edge) for edge in qa_trace.selected_evidence
            ],
            "final_answer": qa_trace.final_answer,
            "trace_path": str(qa_trace_path),
            "retrieval_mode": "kg_question_aware",
        }
        write_json(qa_prediction_path, qa_payload)
        write_json(qa_trace_path, qa_trace.model_dump())

        naive_evidence_preview = "\n".join(
            _edge_to_text(edge) for edge in naive_trace.selected_evidence[:3]
        ) or "-"

        qa_evidence_preview = "\n".join(
            _edge_to_text(edge) for edge in qa_trace.selected_evidence[:3]
        ) or "-"

        per_sample_table = Table(
            title=f"Comparison for {sample.sample_id}",
            show_lines=True,
        )
        per_sample_table.add_column("System", style="cyan")
        per_sample_table.add_column("Contains Gold", justify="center")
        per_sample_table.add_column("Token F1", justify="right")
        per_sample_table.add_column("Prediction", style="green")
        per_sample_table.add_column("Evidence", style="magenta")

        per_sample_table.add_row(
            "baseline",
            _is_contains_gold(baseline_result.final_answer, sample.gold_answer),
            _format_metric(token_overlap_f1(baseline_result.final_answer, sample.gold_answer)),
            baseline_result.final_answer,
            "-",
        )
        per_sample_table.add_row(
            "kg_naive",
            _is_contains_gold(naive_trace.final_answer, sample.gold_answer),
            _format_metric(token_overlap_f1(naive_trace.final_answer, sample.gold_answer)),
            naive_trace.final_answer,
            naive_evidence_preview,
        )
        per_sample_table.add_row(
            "kg_question_aware",
            _is_contains_gold(qa_trace.final_answer, sample.gold_answer),
            _format_metric(token_overlap_f1(qa_trace.final_answer, sample.gold_answer)),
            qa_trace.final_answer,
            qa_evidence_preview,
        )

        console.print(per_sample_table)

        summary_table.add_row(
            sample.sample_id,
            sample.gold_answer or "-",
            baseline_result.final_answer,
            naive_trace.final_answer,
            qa_trace.final_answer,
            naive_evidence_preview,
            qa_evidence_preview,
        )

    console.rule("[bold green]Final Summary")
    console.print(summary_table)
    console.print(f"[bold green]Completed run-all for {len(samples)} sample(s).[/bold green]")