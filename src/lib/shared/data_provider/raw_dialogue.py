from typing import Literal, TypedDict


class RawDialogue(TypedDict):
    idx: int
    role: Literal["seeker", "supporter"]
    content: str