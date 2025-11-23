from typing import TypedDict


class RawSummary(TypedDict):
    idx: int
    capability: str
    question: str
    answer: str
    evidence: list[str]
    theme: str
    group: list[str]