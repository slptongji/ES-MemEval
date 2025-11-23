from typing import Sequence
from langchain_core.messages import BaseMessage, SystemMessage

from lib.shared.prompt_strategies.prompt_strategy import PromptStrategy
from lib.shared.prompt_strategies.session_information import SessionInformation


class FixedStrategy(PromptStrategy):
    def __init__(self, prepend: Sequence[BaseMessage], inplace: str | None, append: Sequence[BaseMessage]) -> None:
        super().__init__()
        self.prepend = list(prepend)
        self.inplace = inplace
        self.append = list(append)
    
    def record_session(self,
                       session_history: Sequence[BaseMessage], 
                       session_information: SessionInformation):
        pass

    def generate_append_prompts(self, 
                                session_history: Sequence[BaseMessage], 
                                session_information: SessionInformation) -> Sequence[BaseMessage]:
        return self.append

    def generate_inplace_prompts(self, 
                                 session_history: Sequence[BaseMessage], 
                                 session_information: SessionInformation) -> str | None:
        return self.inplace

    def generate_prepend_prompts(self, 
                                 session_history: Sequence[BaseMessage], 
                                 session_information: SessionInformation) -> Sequence[BaseMessage]:
        return self.prepend
    