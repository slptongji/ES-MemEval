import abc
import time
from typing import Callable, Sequence
from langchain_core.prompt_values import ChatPromptValue, PromptValue
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from lib.shared.basic_tools.chat_logger import ChatLogger

class ChatModel(abc.ABC):
    def __init__(self, logger: ChatLogger, retry: Sequence[float]) -> None:
        super().__init__()
        self.__logger = logger
        self.__retry = retry

    def _do_with_retry[T](self,    # pylint: disable=un-declared-variable
                          callable: Callable[[], T]):
        delay: float
        for delay in self._retry():
            try:
                return callable()
            except:
                time.sleep(delay)
                pass
        return callable()

    def _retry(self):
        return self.__retry

    @staticmethod
    def map_message_to_itself(message: AIMessage):
        return message

    @staticmethod
    def map_message_to_str(message: AIMessage):
        return message.text()

    def __log_message(self, prompt: ChatPromptValue, message: AIMessage):
        log: list[BaseMessage] = prompt.to_messages()
        log.append(message)
        return self.__logger.log(log)

    def generate[T](self,    # pylint: disable=un-declared-variable
                    messages: Sequence[BaseMessage], 
                    prompt_cache_key: str | None = None,
                    mapping: Callable[[AIMessage], T] = map_message_to_itself) -> tuple[str, T]:
        prompt: ChatPromptValue = ChatPromptValue(messages=messages)

        def _():
            message: AIMessage
            actual_prompt: ChatPromptValue
            message, actual_prompt = self._generate(prompt, prompt_cache_key)
            mapped: T = mapping(message)
            return message, mapped, actual_prompt
        result: tuple[AIMessage, T, ChatPromptValue] = self._do_with_retry(_)

        return self.__log_message(result[2], result[0]), result[1]

    def generate_with_schema[T: BaseModel](self,    # pylint: disable=un-declared-variable
                                           messages: Sequence[BaseMessage], 
                                           schema: type[T], 
                                           prompt_cache_key: str | None = None) -> tuple[str, T]:
        prompt: ChatPromptValue = ChatPromptValue(messages=messages)

        def _():
            return self._generate_with_schema(prompt, schema, prompt_cache_key)
        result: tuple[AIMessage, T, ChatPromptValue] = self._do_with_retry(_)

        return self.__log_message(result[2], result[0]), result[1]
    
    @abc.abstractmethod
    def _generate_with_schema[T: BaseModel](self,    # pylint: disable=un-declared-variable
                                           prompts: ChatPromptValue, 
                                           schema: type[T], 
                                           prompt_cache_key: str | None) -> tuple[AIMessage, T, ChatPromptValue]:
        pass

    @abc.abstractmethod
    def _generate(self, prompts: ChatPromptValue, prompt_cache_key: str | None) -> tuple[AIMessage, ChatPromptValue]:
        pass
    
    @abc.abstractmethod
    def estimate_token_count(self, message: str) -> int:
        pass

    @abc.abstractmethod
    def max_token_count(self) -> int:
        pass