from typing import Literal, Sequence

from lib.shared.basic_tools.chat_logger import ChatLogger
from lib.shared.chat_models.chat_model_indicator import ChatModelIndicator
from lib.shared.chat_models.openai_model import OpenaiModel


class OpenaiModelIndicator(ChatModelIndicator):
    def __init__(self, 
                 model: Literal["gpt-3.5-turbo", "gpt-4o"], 
                 api_key: str, 
                 retry: Sequence[float] = (1, 4, 16, 64)):
        self.__model: Literal["gpt-3.5-turbo", "gpt-4o"] = model
        self.__api_key: str = api_key
        self.__retry: list[float] = list(retry)
    
    def model(self) -> Literal["gpt-3.5-turbo", "gpt-4o"]:
        return self.__model

    def api_key(self) -> str:
        return self.__api_key

    def retry(self) -> Sequence[float]:
        return self.__retry

    def create_model(self, logger: ChatLogger, max_tokens: int | None = None) -> OpenaiModel:
        return OpenaiModel(self.api_key(), logger, self.model(), self.retry(), max_tokens)