from __future__ import annotations


class QuestionClassifier:

    def classify(self, question: str) -> str:
        q = question.lower()

        if any(token in q for token in ["where", "on", "under", "above", "below", "left", "right"]):
            return "spatial"

        if any(token in q for token in ["used for", "use", "purpose"]):
            return "affordance"

        if any(token in q for token in ["made of", "material", "property", "heavy", "fragile"]):
            return "property"

        if any(token in q for token in ["why", "cause", "happen", "fall"]):
            return "causal"

        return "physical_general"