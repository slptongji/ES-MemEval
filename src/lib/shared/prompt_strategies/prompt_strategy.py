import abc
from typing import Literal, Sequence

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage

from lib.shared.prompt_strategies.session_information import SessionInformation


class PromptStrategy(abc.ABC):
    @abc.abstractmethod
    def record_session(self, 
                       session_history: Sequence[BaseMessage], 
                       session_information: SessionInformation):
        pass

    @abc.abstractmethod
    def generate_prepend_prompts(self, 
                                 session_history: Sequence[BaseMessage], 
                                 session_information: SessionInformation) -> Sequence[BaseMessage]:
        pass

    @abc.abstractmethod
    def generate_inplace_prompts(self, 
                                 session_history: Sequence[BaseMessage], 
                                 session_information: SessionInformation) -> str | None:
        pass

    @abc.abstractmethod
    def generate_append_prompts(self, 
                                session_history: Sequence[BaseMessage], 
                                session_information: SessionInformation) -> Sequence[BaseMessage]:
        pass

    def _get_role_of(self, message: BaseMessage) -> Literal["Unknown", "System", "Ai", "Human"]:
        if isinstance(message, SystemMessage):
            return "System"
        if isinstance(message, AIMessage):
            return "Ai"
        if isinstance(message, HumanMessage):
            return "Human"
        return "Unknown"