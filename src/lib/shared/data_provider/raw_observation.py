from typing import TypedDict


class RawObservation(TypedDict):
    idx: int
    utt_id: list[int]
    content: str