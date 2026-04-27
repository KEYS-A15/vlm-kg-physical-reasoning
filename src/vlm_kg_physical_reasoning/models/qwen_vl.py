from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch.nn.functional as F

from vlm_kg_physical_reasoning.models.vlm_spine import VLMBackbone


_ENTITY_LIST_PATTERN = re.compile(r"\[[\s\S]*\]")
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
_FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "answer",
    "best",
    "can",
    "could",
    "does",
    "from",
    "image",
    "in",
    "is",
    "of",
    "question",
    "the",
    "this",
    "what",
    "which",
    "with",
    "would",
}


class QwenVLModel(VLMBackbone):
    """Lazy Qwen2.5-VL wrapper for baseline answering and lightweight extraction."""

    def __init__(
        self,
        model_name: str,
        generation_max_new_tokens: int = 96,
        entity_extraction_max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 0.0,
        device_map: str | None = "auto",
        torch_dtype: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.generation_max_new_tokens = generation_max_new_tokens
        self.entity_extraction_max_new_tokens = entity_extraction_max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.device_map = device_map
        self.torch_dtype = torch_dtype

        self._processor: Any | None = None
        self._model: Qwen2_5_VLForConditionalGeneration | None = None

    def answer(
        self,
        image_path: str,
        question: str,
        evidence: list[str] | None = None,
    ) -> str:
        prompt = self._build_answer_prompt(question=question, evidence=evidence or [])
        return self._generate(image_path=image_path, prompt=prompt, max_new_tokens=self.generation_max_new_tokens)

    def extract_entities(
        self,
        image_path: str,
        question: str,
        max_entities: int,
    ) -> list[str]:
        prompt = (
            "Extract the most concrete physical entities relevant to answering the question. "
            f"Return only a JSON array with at most {max_entities} short lowercase noun phrases. "
            "Do not include attributes, relations, full sentences, or markdown.\n"
            f"Question: {question}"
        )
        raw_output = self._generate(
            image_path=image_path,
            prompt=prompt,
            max_new_tokens=self.entity_extraction_max_new_tokens,
        )
        entities = self._parse_entity_list(raw_output, max_entities=max_entities)
        if entities:
            return entities

        return self._fallback_entities(question=question, max_entities=max_entities)

    # osteofelidae: DEPRECIATED: get embeddings for cosine similarity
    def get_embedding(
        self,
        text: str,
        layer: int = -10,  # TODO NEEDS TUNING
    ) -> torch.Tensor:

        # Load proc + model if not
        processor = self._load_processor()
        model = self._load_model()

        # Calculate inputs
        inputs = self._processor(
            text=[text],
            return_tensors="pt",
        )
        target_device = next(model.parameters()).device
        inputs = inputs.to(target_device)

        # Generate outputs
        with torch.no_grad():
            outputs = self._model(
            **inputs,
            output_hidden_states=True,
        )

        # Extract embedding-like vector (WARNING: JANK)
        hidden = outputs.hidden_states[layer]

        mask = inputs["attention_mask"].unsqueeze(-1).float()
        embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        return F.normalize(embedding.squeeze(0), dim=-1)

    def _generate(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        processor = self._load_processor()
        model = self._load_model()
        image = Image.open(Path(image_path)).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        rendered_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[rendered_prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        target_device = next(model.parameters()).device
        inputs = inputs.to(target_device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature

        generated_ids = model.generate(**inputs, **generation_kwargs)
        trimmed_ids = generated_ids[:, inputs["input_ids"].shape[1] :]

        decoded = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return decoded[0].strip()

    def _load_processor(self) -> Any:
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.model_name)

        return self._processor

    def _load_model(self) -> Qwen2_5_VLForConditionalGeneration:
        if self._model is None:
            load_kwargs: dict[str, Any] = {"torch_dtype": self._resolve_torch_dtype(self.torch_dtype)}
            if self.device_map:
                load_kwargs["device_map"] = self.device_map

            try:
                self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    **load_kwargs,
                )
            except (ImportError, ValueError):
                load_kwargs.pop("device_map", None)
                self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    **load_kwargs,
                )

        return self._model

    @staticmethod
    def _resolve_torch_dtype(value: str) -> torch.dtype | str:
        if value == "auto":
            return value
        dtype = getattr(torch, value, None)
        if dtype is None:
            raise ValueError(f"Unsupported torch dtype: {value}")
        return dtype

    @staticmethod
    def _build_answer_prompt(question: str, evidence: list[str]) -> str:
        if not evidence:
            return (
                "Answer the question about the image as clearly and briefly as possible.\n"
                f"Question: {question}"
            )

        evidence_block = "\n".join(f"- {item}" for item in evidence)
        return (
            "Use the image and the compact knowledge graph evidence below to answer the question. "
            "If the evidence is noisy, prefer the image.\n"
            f"Question: {question}\n"
            "Evidence:\n"
            f"{evidence_block}"
        )

    def _parse_entity_list(self, raw_output: str, max_entities: int) -> list[str]:
        candidate_text = raw_output.strip()
        match = _ENTITY_LIST_PATTERN.search(candidate_text)
        if match is not None:
            candidate_text = match.group(0)

        try:
            parsed = json.loads(candidate_text)
        except json.JSONDecodeError:
            return []

        if not isinstance(parsed, list):
            return []

        entities: list[str] = []
        seen: set[str] = set()
        for item in parsed:
            if not isinstance(item, str):
                continue
            normalized = self._normalize_entity(item)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            entities.append(normalized)
            if len(entities) >= max_entities:
                break

        return entities

    def _fallback_entities(self, question: str, max_entities: int) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()

        for token in _TOKEN_PATTERN.findall(question.lower()):
            if token in _FALLBACK_STOPWORDS or len(token) < 3:
                continue
            normalized = self._normalize_entity(token)
            if normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(normalized)
            if len(candidates) >= max_entities:
                break

        return candidates or ["object"]

    @staticmethod
    def _normalize_entity(value: str) -> str:
        cleaned = re.sub(r"[^a-z0-9\s]+", " ", value.lower()).strip()
        return re.sub(r"\s+", " ", cleaned)
