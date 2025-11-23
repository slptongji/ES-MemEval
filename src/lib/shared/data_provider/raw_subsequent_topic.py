from typing import TypedDict


class RawSubsequentTopic(TypedDict):
    idx: int
    topic: str
    psychological_condition: str
    physical_condition: str
    more_details: str
    related_sessions: list[str]