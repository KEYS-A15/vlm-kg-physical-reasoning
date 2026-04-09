from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    stage: str


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_data: str = ""
    trace_output_dir: str
    prediction_output_dir: str


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_entities: int = Field(default=5, ge=1)
    max_evidence_triples: int = Field(default=5, ge=1)


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str = "INFO"


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    generation_max_new_tokens: int = Field(default=96, ge=1)
    entity_extraction_max_new_tokens: int = Field(default=64, ge=1)
    do_sample: bool = False
    temperature: float = Field(default=0.0, ge=0.0)
    device_map: str | None = "auto"
    torch_dtype: str = "auto"


class ConceptNetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = "https://api.conceptnet.io"
    timeout_seconds: float = Field(default=10.0, gt=0.0)
    max_edges_per_node: int = Field(default=10, ge=1)
    language: str = "en"


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str = "conceptnet_naive"
    conceptnet: ConceptNetConfig = Field(default_factory=ConceptNetConfig)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: ProjectConfig
    paths: PathsConfig
    pipeline: PipelineConfig
    logging: LoggingConfig
    model: ModelConfig
    retrieval: RetrievalConfig


def load_config(config_path: str | Path = "configs/default.yaml") -> AppConfig:
    """Load the project configuration from YAML."""

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    return AppConfig.model_validate(raw)
