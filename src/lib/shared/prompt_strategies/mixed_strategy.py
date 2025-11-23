from typing import Sequence
from langchain_core.messages import BaseMessage, SystemMessage

from lib.shared.prompt_strategies.prompt_strategy import PromptStrategy
from lib.shared.prompt_strategies.session_information import SessionInformation


class MixedStrategy(PromptStrategy):
    def __init__(self, strategy_before: PromptStrategy, strategy_after: PromptStrategy) -> None:
        super().__init__()
        self.strategy_before: PromptStrategy = strategy_before
        self.strategy_after: PromptStrategy = strategy_after
    
    def record_session(self,
                       session_history: Sequence[BaseMessage], 
                       session_information: SessionInformation):
        self.strategy_before.record_session(session_history, session_information)
        self.strategy_after.record_session(session_history, session_information)

    def generate_append_prompts(self, 
                                session_history: Sequence[BaseMessage], 
                                session_information: SessionInformation) -> Sequence[BaseMessage]:
        result: list[BaseMessage] = []
        result.extend(self.strategy_before.generate_append_prompts(session_history, session_information))
        result.extend(self.strategy_after.generate_append_prompts(session_history, session_information))
        return result

    def generate_inplace_prompts(self, 
                                 session_history: Sequence[BaseMessage], 
                                 session_information: SessionInformation) -> str | None:
        before: str | None = self.strategy_before.generate_inplace_prompts(session_history, session_information)
        after: str | None = self.strategy_after.generate_inplace_prompts(session_history, session_information)

        if before is None:
            return after
        if after is None:
            return before
        return f"{before}\n{after}"
            

    def generate_prepend_prompts(self, 
                                 session_history: Sequence[BaseMessage], 
                                 session_information: SessionInformation) -> Sequence[BaseMessage]:
        result: list[BaseMessage] = []
        result.extend(self.strategy_before.generate_prepend_prompts(session_history, session_information))
        result.extend(self.strategy_after.generate_prepend_prompts(session_history, session_information))
        return result
    