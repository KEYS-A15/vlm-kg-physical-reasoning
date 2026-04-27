from __future__ import annotations
from pathlib import Path
from vlm_kg_physical_reasoning.utils.io import write_json, load_json
from sentence_transformers import SentenceTransformer
import torch

question_mappings = {
    "spatial":    ["space", "location", "position"],
    "affordance": ["purpose", "used for", "function"],
    "property":   ["property", "material", "weight"],
    "causal":     ["cause", "reason", "why"],
}


# osteofelidae: improved question classifier
class QuestionClassifier:

    def __init__(
        self,
        classify_mode: str = "cosine",
        cosine_thres: float = 0.1,
        embed_path: Path = Path("embeds/embeds.json"),  # TODO make this configurable
        embedding_model_name: str = "all-MiniLM-L6-v2",  # TODO make this configurable
    ) -> None:

        # Check classify mode
        if classify_mode not in ("static", "cosine"):
            raise ValueError(f"classify_mode must be 'static' or 'cosine', got {classify_mode}")

        # Set instance vars
        self._classify_mode = classify_mode
        self._cosine_thres = cosine_thres
        self._embed_path = embed_path

        # Load embedding model if cosine method
        if classify_mode == "cosine":
            self._embed_model = SentenceTransformer(embedding_model_name)

        # Embedding cache
        self._embed_cache: dict[str, torch.Tensor] | None = None

    # Get embedding from model
    def _get_embedding(self, text: str) -> torch.Tensor:
        vec = self._embed_model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        return vec

    # Load embeddings from file, or return if already loaded
    def _load_embeddings(self) -> dict[str, torch.Tensor]:

        # Return if loaded
        if self._embed_cache is not None:
            return self._embed_cache

        # Precompute if not already
        if not self._embed_path.exists():
            print("Precomputing embeds...")
            self._embed_path.parent.mkdir(parents=True, exist_ok=True)
            embed_dict = {
                word: self._get_embedding(word).tolist()
                for keywords in question_mappings.values()
                for word in keywords
            }
            write_json(self._embed_path, embed_dict)

        # Load if computed
        else:
            embed_dict = load_json(self._embed_path)
            print("Loaded embeds successfully")

        # Turn to tensors
        device = self._embed_model.device
        self._embed_cache = {
            word: torch.tensor(vec, device=device) for word, vec in embed_dict.items()
        }

        return self._embed_cache

    # Cosine classify
    def _cosine_classify(self, question: str) -> str:
        embed_dict = self._load_embeddings()
        question_embed = self._get_embedding(question)

        cosi = torch.nn.CosineSimilarity(dim=-1)

        # Reverse mapping
        word_to_category = {
            word: category
            for category, keywords in question_mappings.items()
            for word in keywords
        }

        # Score each category based on keywords
        category_scores: dict[str, float] = {cat: 0.0 for cat in question_mappings}
        for word, kw_embed in embed_dict.items():
            score = cosi(kw_embed, question_embed).item()
            category = word_to_category[word]
            category_scores[category] = max(category_scores[category], score)

        best_category = max(category_scores, key=category_scores.__getitem__)
        best_score = category_scores[best_category]

        print(category_scores)

        # If insufficient score, return default
        if best_score < self._cosine_thres:
            return "physical_general"
        return best_category

    # Static classify
    def _static_classify(self, question: str) -> str:
        q = question.lower()

        # Changed this to regex to stop matching keywords in other unrelated words
        checks = [
            ("spatial",   r"\b(where|on|under|above|below|left|right)\b"),
            ("affordance", r"\b(used for|use|purpose)\b"),
            ("property",  r"\b(made of|material|property|heavy|fragile)\b"),
            ("causal",    r"\b(why|cause|happen|fall)\b"),
        ]

        import re
        for category, pattern in checks:
            if re.search(pattern, q):
                return category

        return "physical_general"

    # Overall classify
    def classify(self, question: str) -> str:

        if self._classify_mode == "cosine":
            ans = self._cosine_classify(question)
        else:
            ans = self._static_classify(question)
        print(f"Classified: {ans}")
        return ans