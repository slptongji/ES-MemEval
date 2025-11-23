from typing import NamedTuple, Sequence

from lib.shared.basic_tools.chat_logger import ChatLogger
from lib.shared.chat_models.chat_model_indicator import ChatModelIndicator
from lib.shared.chat_models.vllm_human_ai_only_model import VllmHumanAiOnlyModel
from lib.shared.chat_models.vllm_model import VllmModel
from lib.shared.chat_models.vllm_model_indicator import VllmModelIndicator


class VllmHumanAiOnlyModelIndicator(ChatModelIndicator):
    def __init__(self, 
                 url: str, 
                 model: str, 
                 api_key: str, 
                 max_tokens: int, 
                 retry: Sequence[float] = (1, 2, 4, 8)):
        self.__url: str = url
        self.__model: str = model
        self.__api_key: str = api_key
        self.__max_tokens: int = max_tokens
        self.__retry: list[float] = list(retry)
    
    def url(self) -> str:
        return self.__url

    def model(self) -> str:
        return self.__model

    def api_key(self) -> str:
        return self.__api_key
    
    def max_tokens(self) -> int:
        return self.__max_tokens

    def retry(self) -> Sequence[float]:
        return self.__retry

    def create_model(self, logger: ChatLogger, max_tokens: int | None = None) -> VllmHumanAiOnlyModel:
        if max_tokens is None or max_tokens > self.max_tokens():
            max_tokens = self.max_tokens()
        return VllmHumanAiOnlyModel(self.url(), self.model(), self.api_key(), 
                                    logger, self.retry(), max_tokens)
    
    def as_normal_vllm(self) -> VllmModelIndicator:
        return VllmModelIndicator(self.url(), 
                                  self.model(), 
                                  self.api_key(), 
                                  self.max_tokens(), 
                                  self.retry())
