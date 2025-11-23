import re
from typing import Any, Sequence

from lib.shared.metrics.metric import Metric


class F1Score(Metric[str, str, Any]):
    def compute(self, gold: str, prediction: str, more: Any = None):
        gold_tokens: list[str] = self.normalize_text(gold).split()
        prediction_tokens: list[str] = self.normalize_text(prediction).split()
        common: set = set(gold_tokens) & set(prediction_tokens)
        precision: float = len(common) / len(prediction_tokens)
        recall: float = len(common) / len(gold_tokens)
        f1: float = 0 if len(common) == 0 else (2 * (precision * recall) / (precision + recall))
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def normalize_text(self, text: str):
        return re.sub(r'\W+', ' ', text.lower()).strip()
    
    def submetric_keys(self) -> Sequence[str]:
        return ["precision", "recall", "f1_score"]