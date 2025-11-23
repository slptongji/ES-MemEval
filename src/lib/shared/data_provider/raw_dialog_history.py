from typing import TypedDict

from lib.shared.data_provider.raw_dialogue import RawDialogue
from lib.shared.data_provider.raw_observation import RawObservation


class RawDialogHistory(TypedDict):
    id: str
    timestamp: str
    emotion: str
    topic: str
    summary: str
    dialogue: list[RawDialogue]
    observation: list[RawObservation]