import abc
from typing import Mapping, Sequence

class Metric[TGold, TPrediction, TMoreInformation](abc.ABC):    # pylint: disable=un-declared-variable
    @abc.abstractmethod
    def submetric_keys(self) -> Sequence[str]:
        pass

    @abc.abstractmethod
    def compute(self, gold: TGold, prediction: TPrediction, more: TMoreInformation) -> Mapping[str, float | str]:
        pass