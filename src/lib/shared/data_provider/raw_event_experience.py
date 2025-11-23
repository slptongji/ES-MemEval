from typing import TypedDict


class RawEventExperience(TypedDict):
    id: str
    date: str
    conv_id: str
    event: str
    influenced_by: list[str]