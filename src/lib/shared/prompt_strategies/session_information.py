from typing import Sequence, TypedDict


class SessionInformation(TypedDict):
    date: str | None
    human_name: str | None
    ai_name: str | None
    message_indices: Sequence[str | None] | None
    