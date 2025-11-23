from typing import TypedDict

from lib.shared.data_provider.raw_question import RawQuestion


class RawQuestionGroup(TypedDict):
    id: str
    questions: list[RawQuestion]