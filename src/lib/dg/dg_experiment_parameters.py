from pathlib import Path
from typing import Callable, NamedTuple

from langchain_core.messages import SystemMessage

from lib.shared.chat_models.chat_model_indicator import ChatModelIndicator
from lib.shared.chat_rooms.chat_room import ChatRoom
from lib.shared.data_provider.raw_seeker import RawSeeker


class DgExperimentParameters(NamedTuple):
    output_directory: Path
    seeker: ChatModelIndicator
    supporter: ChatModelIndicator
    observation_scorer: ChatModelIndicator
    observation_usage_judge: ChatModelIndicator
    overall_scorer: ChatModelIndicator

    class SupporterChatRoomParameters(NamedTuple):
        data: RawSeeker
        beginning_prompt: SystemMessage
        prefer_cuda: bool

    supporter_chat_room: Callable[[SupporterChatRoomParameters], ChatRoom]
    prefer_cuda: bool
    skip_turn_observation_wise: bool