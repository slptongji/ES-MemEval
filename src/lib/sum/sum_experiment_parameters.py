from pathlib import Path
from typing import Callable, NamedTuple

from langchain_core.messages import SystemMessage

from lib.shared.chat_models.chat_model import ChatModel
from lib.shared.chat_models.chat_model_indicator import ChatModelIndicator
from lib.shared.chat_rooms.chat_room import ChatRoom
from lib.shared.data_provider.raw_seeker import RawSeeker
from lib.shared.document_stores.document_store import DocumentStore
from lib.shared.prompt_strategies.prompt_strategy import PromptStrategy


class SumExperimentParameters(NamedTuple):
    output_directory: Path

    model: ChatModelIndicator
    context_length: int
    judge: ChatModelIndicator

    class MemoryStrategyParameters(NamedTuple):
        prefer_cuda: bool

    memory_strategy: Callable[[MemoryStrategyParameters], PromptStrategy]
    prefer_cuda: bool