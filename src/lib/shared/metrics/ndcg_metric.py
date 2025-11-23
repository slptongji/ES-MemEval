import math
from typing import AbstractSet, Any, Generic, Sequence

from lib.shared.metrics.metric import Metric


class NdcgMetric(Metric[AbstractSet[str], Sequence[str], Any]):
    def __init__(self, candidate_size: int) -> None:
        super().__init__()
        self.candidate_size = candidate_size

    def submetric_keys(self) -> Sequence[str]:
        return ["dcg", "idcg", "ndcg"]

    @staticmethod
    def dcg(gold: AbstractSet[str], prediction: Sequence[str]):
        result: float = 0.0
        index: int
        retrieved: str
        for index, retrieved in enumerate(prediction, 2):
            if retrieved in gold:
                result += 1 / math.log2(index + 2)
        return result

    @staticmethod
    def idcg(k: int, gold: AbstractSet[str]):
        return sum(1 / math.log2(i) for i in range(2, min(len(gold), k) + 2))

    def compute(self, gold: AbstractSet[str], prediction: Sequence[str], more: Any):
        if len(prediction) > self.candidate_size:
            prediction = prediction[:self.candidate_size]
        dcg: float = self.dcg(gold, prediction)
        idcg: float = self.idcg(self.candidate_size, gold)
        ndcg: float = dcg / idcg if idcg > 0 else 0
        return {
            "dcg": dcg,
            "idcg": idcg,
            "ndcg": ndcg
        }