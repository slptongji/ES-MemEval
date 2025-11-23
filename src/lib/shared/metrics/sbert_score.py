from typing import Any, Sequence
from langchain_core.embeddings import Embeddings
import sentence_transformers
import sentence_transformers.util
from torch import Tensor
from lib.shared.metrics.metric import Metric


class SbertScore(Metric[str, str, Any]):
    def __init__(self, model: Embeddings) -> None:
        super().__init__()
        self.model = model

    def compute(self, gold: str, prediction: str, more: Any = None):
        gold_embedding: list[float] = self.model.embed_query(gold)
        prediction_embedding: list[float] = self.model.embed_query(prediction)
        result: Tensor = sentence_transformers.util.cos_sim(gold_embedding, prediction_embedding)
        return {
            "value": float(result)
        }
    
    def submetric_keys(self) -> Sequence[str]:
        return ["value"]