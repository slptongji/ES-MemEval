from pathlib import Path
import re
from typing import Callable, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompt_values import PromptValue

from lib.shared.basic_tools.chat_logger import ChatLogger
from lib.shared.chat_models.chat_model import ChatModel
from lib.shared.prompt_strategies.prompt_strategy import PromptStrategy
from lib.shared.prompt_strategies.session_information import SessionInformation


class ChatRoom:
    @staticmethod
    def default_inplace_behavior(messages: Sequence[BaseMessage], inplace_prompt: str | None):
        if inplace_prompt is None:
            return messages
        raise ValueError("It is uncertain how inplace prompts are joined. You should specify the inplace behavior manually.")

    def __init__(self, 
                 prompt: PromptStrategy, 
                 prompt_cache_key: str | None = None, 
                 inplace_behavior: Callable[[Sequence[BaseMessage], str | None], Sequence[BaseMessage]] = default_inplace_behavior) -> None:
        self.prompt: PromptStrategy = prompt
        self._history: list[BaseMessage] = []
        self.session: SessionInformation | None = None
        self.prompt_cache_key = prompt_cache_key
        self.inplace_behavior = inplace_behavior

    def begin_session(self, information: SessionInformation | None = None):
        if information is None:
            information = {"ai_name": None, "date": None, "human_name": None, "message_indices": None}
        if self.session is not None:
            raise RuntimeError("It's currently in a session. Please use end_session or drop_session before starting the next session.")
        self._history = []
        self.session = information

    def append(self, message: BaseMessage) -> None:
        if self.session is None:
            raise RuntimeError("It's not in a session now. Please call begin_session before starting to chat.")
        self._history.append(message)

    def session_history(self) -> Sequence[BaseMessage]:
        return list(self._history)

    def generate(self, model: ChatModel) -> tuple[str, AIMessage]:
        result: tuple[str, AIMessage] = self.generate_without_append(model)
        self.append(result[1])
        return result
    
    def generate_without_append(self, model: ChatModel) -> tuple[str, AIMessage]:
        if self.session is None:
            raise RuntimeError("It's not in a session now. Please call begin_session before starting to chat.")
        
        prompts: list[BaseMessage] = []
        prompts.extend(self.prompt.generate_prepend_prompts(self._history, self.session))
        prompts.extend(self.inplace_behavior(self._history, 
                                             self.prompt.generate_inplace_prompts(self._history, self.session)))
        prompts.extend(self.prompt.generate_append_prompts(self._history, self.session))
        
        result: tuple[str, AIMessage] = model.generate(prompts, self.prompt_cache_key)
        return result
    
    def generate_str(self, model: ChatModel) -> tuple[str, str]:
        result: tuple[str, BaseMessage] = self.generate(model)
        return result[0], result[1].text()
    
    def end_session(self):
        if self.session is None:
            raise RuntimeError("It's not in a session now.")
        
        self.prompt.record_session(self._history, self.session)
        self.session = None

    def drop_session(self):
        if self.session is None:
            raise RuntimeError("It's not in a session now.")
        self.session = None
