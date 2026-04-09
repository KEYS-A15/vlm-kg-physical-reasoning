# RKG-VLM
### Retrieval-Enhanced Knowledge Graph Reasoning for Vision-Language Models

Minimal class-project pipeline for improving VLM physical-world reasoning with compact ConceptNet evidence.

## Current Phase
- Baseline VLM inference is implemented with a real Qwen2.5-VL wrapper.
- Naive KG augmentation is implemented with:
  - VLM-based entity extraction
  - simple ConceptNet node mapping
  - 1-hop ConceptNet retrieval
  - naive top-k evidence selection
  - JSON prediction and trace outputs

## Setup
```bash
uv sync
```

## Commands
Run the direct baseline:

```bash
uv run vlm-kg-physical-reasoning run-baseline --config configs/default.yaml --sample-file path/to/sample.json
```

Run the naive KG pipeline:

```bash
uv run vlm-kg-physical-reasoning run-kg-naive --config configs/default.yaml --sample-file path/to/sample.json
```

Use a smaller checkpoint if needed:

```bash
uv run vlm-kg-physical-reasoning run-kg-naive --config configs/default.yaml --sample-file path/to/sample.json --model-name Qwen/Qwen2.5-VL-3B-Instruct
```

## Sample File Format
The sample file can contain a single object, a list, or `{"samples": [...]}`:

```json
{
  "sample_id": "sample-001",
  "image_path": "path/to/image.jpg",
  "question": "What happens if the cup is near the edge of the table?"
}
```

## Notes
- This phase keeps retrieval intentionally naive.
- ConceptNet failures are non-fatal and are recorded in the trace output.
- The next phase can build on this structure for question-aware retrieval, relation filtering, and reranking.
