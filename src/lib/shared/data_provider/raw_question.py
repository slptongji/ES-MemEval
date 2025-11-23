from typing import TypedDict


class RawQuestion(TypedDict):
    idx: int
    capability: str
    question: str
    answer: str
    evidence: list[str]