import abc
from typing import Sequence

from lib.shared.basic_tools.chat_logger import ChatLogger
from lib.shared.chat_models.chat_model import ChatModel


class ChatModelIndicator(abc.ABC):
    @abc.abstractmethod
    def create_model(self, 
                     logger: ChatLogger, 
                     max_tokens: int | None = None) -> ChatModel:
        pass
