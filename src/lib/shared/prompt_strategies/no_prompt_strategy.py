from typing import Sequence
from langchain_core.messages import BaseMessage

from lib.shared.prompt_strategies.fixed_strategy import FixedStrategy
from lib.shared.prompt_strategies.prompt_strategy import PromptStrategy
from lib.shared.prompt_strategies.session_information import SessionInformation


class NoPromptStrategy(FixedStrategy):
    def __init__(self) -> None:
        super().__init__([], None, [])
    