from typing import TypedDict

from lib.shared.data_provider.raw_basic_info import RawBasicInfo
from lib.shared.data_provider.raw_dialog_history import RawDialogHistory
from lib.shared.data_provider.raw_event_experience import RawEventExperience
from lib.shared.data_provider.raw_question_group import RawQuestionGroup
from lib.shared.data_provider.raw_subsequent_topic import RawSubsequentTopic
from lib.shared.data_provider.raw_summary import RawSummary


class RawSeeker(TypedDict):
    id: str
    basic_info: RawBasicInfo
    event_experience: list[RawEventExperience]
    social_relationship: list[str]
    dialog_history: list[RawDialogHistory]
    questions: list[RawQuestionGroup]
    summaries: list[RawSummary]
    subsequent_topics: list[RawSubsequentTopic]
    