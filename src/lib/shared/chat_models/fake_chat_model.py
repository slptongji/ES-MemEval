from pathlib import Path
import random
import re
import time
import typing
import langchain_core
from langchain_core.language_models import LanguageModelInput
import langchain_core.messages
import langchain_core.messages.utils
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_core.prompt_values import ChatPromptValue, PromptValue
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from openai import BadRequestError, RateLimitError
from pydantic import BaseModel, Field, SecretStr
import pydantic_rng
import tiktoken

from lib.shared.basic_tools.chat_logger import ChatLogger
from lib.shared.basic_tools.pydantic_random import PydanticRandom
from lib.shared.basic_tools.string_random import StringRandom
from lib.shared.chat_models.chat_model import ChatModel


class FakeChatModel(ChatModel):
    def __init__(self, logger: ChatLogger) -> None:
        super().__init__(logger, ())

    def _generate_with_schema[T: BaseModel](self,    # pylint: disable=un-declared-variable
                                            prompts: ChatPromptValue, 
                                            schema: type[T], 
                                            prompt_cache_key: str | None) -> tuple[AIMessage, T, ChatPromptValue]:
        seed: str = "FIRST MESSAGE"
        if len(prompts.messages) > 0:
            seed: str = prompts.messages[-1].text()
        result = PydanticRandom(random.Random(seed)).generate(schema)
        message = AIMessage(result.model_dump_json())
        return message, result, prompts

    def _generate(self, prompts: ChatPromptValue, prompt_cache_key: str | None) -> tuple[AIMessage, ChatPromptValue]:
        seed: str = "FIRST MESSAGE"
        if len(prompts.messages) > 0:
            seed: str = prompts.messages[-1].text()
        result = StringRandom(random.Random(seed), 10, True).next()
        message = AIMessage(result)
        return message, prompts
    
    def estimate_token_count(self, message: str) -> int:
        raise NotImplementedError()

    def max_token_count(self) -> int:
        raise NotImplementedError()
