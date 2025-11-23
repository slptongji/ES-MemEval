from typing import AbstractSet, Any, Sequence

from lib.shared.metrics.metric import Metric


class RecallMetric(Metric[AbstractSet[str], Sequence[str], Any]):
    def __init__(self, candidate_size: int) -> None:
        super().__init__()
        self.candidate_size = candidate_size

    def submetric_keys(self) -> Sequence[str]:
        return ["value"]

    def compute(self, gold: AbstractSet[str], prediction: Sequence[str], more: Any):
        if len(prediction) > self.candidate_size:
            prediction = prediction[:self.candidate_size]
        hit: set = set(prediction) & set(gold)
        return {
            "value": len(hit) / len(gold) if len(gold) != 0 else 0.0
        }