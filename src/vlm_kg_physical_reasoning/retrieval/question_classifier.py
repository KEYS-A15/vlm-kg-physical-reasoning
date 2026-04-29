from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from vlm_kg_physical_reasoning.utils.io import load_json, write_json
from vlm_kg_physical_reasoning.utils.logging import get_logger

logger = get_logger(__name__)


question_mappings = {
    "spatial": ["space", "location", "position"],
    "affordance": ["purpose", "used for", "function"],
    "property": ["property", "material", "weight"],
    "causal": ["cause", "reason", "why"],
}


class QuestionClassifier:
    def __init__(
        self,
        classify_mode: str = "static",
        cosine_thres: float = 0.1,
        embed_path: Path = Path("embeds/embeds.json"),
        embedding_model_name: str = "all-MiniLM-L6-v2",
        verbose: bool = False,
    ) -> None:
        if classify_mode not in ("static", "cosine"):
            raise ValueError(
                f"classify_mode must be 'static' or 'cosine', got {classify_mode}"
            )

        self._classify_mode = classify_mode
        self._cosine_thres = cosine_thres
        self._embed_path = embed_path
        self._verbose = verbose

        self._embed_model: SentenceTransformer | None = None
        if classify_mode == "cosine":
            self._embed_model = SentenceTransformer(embedding_model_name)

        self._embed_cache: dict[str, torch.Tensor] | None = None

    def _require_embed_model(self) -> SentenceTransformer:
        if self._embed_model is None:
            raise RuntimeError(
                "Embedding model is not loaded because classify_mode is not 'cosine'."
            )
        return self._embed_model

    def _get_embedding(self, text: str) -> torch.Tensor:
        model = self._require_embed_model()
        vec = model.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        if not isinstance(vec, torch.Tensor):
            raise TypeError("SentenceTransformer returned a non-tensor embedding.")

        return vec

    def _load_embeddings(self) -> dict[str, torch.Tensor]:
        if self._embed_cache is not None:
            return self._embed_cache

        model = self._require_embed_model()

        if not self._embed_path.exists():
            if self._verbose:
                logger.info("Precomputing question-classifier embeddings at %s", self._embed_path)

            self._embed_path.parent.mkdir(parents=True, exist_ok=True)
            raw_embed_dict: dict[str, list[float]] = {
                word: self._get_embedding(word).detach().cpu().tolist()
                for keywords in question_mappings.values()
                for word in keywords
            }
            write_json(self._embed_path, raw_embed_dict)
        else:
            raw_loaded: Any = load_json(self._embed_path)
            if not isinstance(raw_loaded, dict):
                raise ValueError(f"Embedding cache must be a dict: {self._embed_path}")

            raw_embed_dict = {}
            for word, vec in raw_loaded.items():
                if isinstance(word, str) and isinstance(vec, list):
                    raw_embed_dict[word] = [float(x) for x in vec]

            if self._verbose:
                logger.info("Loaded question-classifier embeddings from %s", self._embed_path)

        device = model.device
        embed_cache: dict[str, torch.Tensor] = {
            word: torch.tensor(vec, device=device)
            for word, vec in raw_embed_dict.items()
        }

        self._embed_cache = embed_cache
        return embed_cache

    def _cosine_classify(self, question: str) -> str:
        embed_dict = self._load_embeddings()
        question_embed = self._get_embedding(question)

        cosi = torch.nn.CosineSimilarity(dim=-1)

        word_to_category = {
            word: category
            for category, keywords in question_mappings.items()
            for word in keywords
        }

        category_scores: dict[str, float] = {cat: 0.0 for cat in question_mappings}
        for word, kw_embed in embed_dict.items():
            score = cosi(kw_embed, question_embed).item()
            category = word_to_category[word]
            category_scores[category] = max(category_scores[category], score)

        best_category = max(category_scores, key=category_scores.__getitem__)
        best_score = category_scores[best_category]

        if self._verbose:
            logger.debug("Question category scores: %s", category_scores)
            logger.info("Classified question_type=%s", best_category)

        if best_score < self._cosine_thres:
            return "physical_general"

        return best_category

    def _static_classify(self, question: str) -> str:
        q = question.lower()

        checks = [
            ("spatial", r"\b(where|on|under|above|below|left|right|front|behind|between)\b"),
            ("affordance", r"\b(used for|use|purpose|can.*do|function)\b"),
            ("property", r"\b(made of|material|property|heavy|fragile|soft|hard|liquid)\b"),
            ("causal", r"\b(why|cause|happen|fall|spill|break)\b"),
        ]

        for category, pattern in checks:
            if re.search(pattern, q):
                return category

        return "physical_general"

    def classify(self, question: str) -> str:
        if self._classify_mode == "cosine":
            return self._cosine_classify(question)

        return self._static_classify(question)