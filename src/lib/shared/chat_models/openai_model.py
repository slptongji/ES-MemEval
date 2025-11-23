from pathlib import Path
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
import tiktoken

from lib.shared.basic_tools.chat_logger import ChatLogger
from lib.shared.chat_models.chat_model import ChatModel


class OpenaiModel(ChatModel):
    context_windows: dict[str, int] = {
        "gpt-3.5-turbo": 16385,
        "gpt-4o": 128000
    }

    def __init__(self, api_key: str, logger: ChatLogger, 
                 model: typing.Literal["gpt-3.5-turbo", "gpt-4o"], 
                 retry: typing.Sequence[float], 
                 max_tokens: int | None) -> None:
        super().__init__(logger, retry)
        self.model_name = model
        self.client = ChatOpenAI(api_key=SecretStr(api_key),     # pyright: ignore[reportCallIssue]
                                 model=model,     # pyright: ignore[reportCallIssue]
                                 max_completion_tokens=max_tokens) # pyright: ignore[reportCallIssue]
        self.tiktoken: tiktoken.Encoding | None = None

    def _generate_with_schema[T: BaseModel](self,    # pylint: disable=un-declared-variable
                                            prompts: ChatPromptValue, 
                                            schema: type[T], 
                                            prompt_cache_key: str | None) -> tuple[AIMessage, T, ChatPromptValue]:
        kwargs: dict[str, typing.Any] = {}
        if prompt_cache_key is not None:
            kwargs["prompt_cache_key"] = prompt_cache_key

        client: Runnable = self.client.with_structured_output(schema, include_raw=True, strict=True)
        def invoke_and_raise() -> tuple[AIMessage, T]:
            class StructuredOutput(typing.TypedDict):
                raw: AIMessage
                parsed: T | None
                parsing_error: BaseException | None
            result: StructuredOutput = client.invoke(prompts, **kwargs)
            if result["parsing_error"] is not None:
                raise typing.cast(BaseException, result["parsing_error"])
            return result["raw"], typing.cast(T, result["parsed"])

        error: RateLimitError
        try:
            result: tuple[AIMessage, T] = invoke_and_raise()
        except RateLimitError as error:
            delay: float | None = None

            match: re.Match[str] | None = re.search(r"Please try again in (\d+)ms.", error.message)
            if match is not None:
                delay = float(match.group(1)) / 1000
            
            if delay is None:
                match = re.search(r"Please try again in (\d+)s.", error.message)
                if match is not None:
                    delay = float(match.group(1))

            if delay is None:
                raise

            time.sleep(delay + 0.1)
            result = invoke_and_raise()

        return result[0], result[1], prompts

    def _generate(self, prompts: ChatPromptValue, prompt_cache_key: str | None) -> tuple[AIMessage, ChatPromptValue]:
        kwargs: dict[str, typing.Any] = {}
        if prompt_cache_key is not None:
            kwargs["prompt_cache_key"] = prompt_cache_key

        error: BaseException
        try:
            result: BaseMessage = self.client.invoke(prompts, **kwargs)
        except RateLimitError as error:
            delay: float | None = None

            match: re.Match[str] | None = re.search(r"Please try again in (\d+(\.\d+)?)ms.", error.message)
            if match is not None:
                delay = float(match.group(1)) / 1000
            
            if delay is None:
                match = re.search(r"Please try again in (\d+(\.\d+)?)s.", error.message)
                if match is not None:
                    delay = float(match.group(1))

            if delay is None:
                raise

            time.sleep(delay + 0.1)
            result = self.client.invoke(prompts, **kwargs)
        except BadRequestError as error:
            rest: int | None = None

            match = re.search(
                r"This model's maximum context length is (\d+) tokens. However, you requested (\d+) tokens", 
                error.message)
            if match is not None:
                rest = int(match.group(2)) - int(match.group(1))

            if rest is None:
                match = re.search(
                    r"This model's maximum context length is (\d+) tokens. "
                    r"However, your messages resulted in (\d+) tokens.",
                    error.message)
                if match is not None:
                    rest = int(match.group(2)) - int(match.group(1))

            if rest is None:
                raise

            new_messages: list[BaseMessage] = prompts.to_messages()
            while rest > 0:
                rest -= self.estimate_token_count(new_messages.pop(0).text())
            return self._generate(ChatPromptValue(messages=new_messages), prompt_cache_key)

        assert isinstance(result, AIMessage)
        return result, prompts
    
    def estimate_token_count(self, message: str) -> int:
        if self.tiktoken is None:
            self.tiktoken = self._do_with_retry(lambda: tiktoken.encoding_for_model(self.model_name))
        return len(self.tiktoken.encode(message))

    def max_token_count(self) -> int:
        return OpenaiModel.context_windows[self.model_name]


def _test():
    model: OpenaiModel = OpenaiModel(input("API KEY: "), 
                                     ChatLogger(Path("./outputs/test/openai_model")), 
                                     "gpt-4o", (), None)
    print(model.generate([
        SystemMessage("This is an ability test. You should just repeat the input messages without modification."),
        HumanMessage("Lorem ipsum dolor sit amet, consectetur adipiscing elit.")
    ])[1].text())

    print(model.generate([
        HumanMessage("Lorem ipsum dolor sit amet."),
        AIMessage("Sorry but I can't understand what are you saying."),
        HumanMessage("Lorem means hello."),
        AIMessage("But what does \"ipsum dolor sit amet\" mean?"),
        HumanMessage("Please guess."),
        AIMessage("I'm sorry but I can't guess as I am an AI and I can only provide factual information."),
        HumanMessage("Oh! I see. But I don't want any information from you. It's just a game between you and me."),
    ])[1].text())

    class WeatherFormatter(BaseModel):
        weather: str = Field(description="weather")
        wetness: int = Field(description="wetness")
        temperature: int = Field(description="temperature")

    print(model.generate_with_schema([
        HumanMessage("Hows the weather today?")
    ], schema=WeatherFormatter))


if __name__ == "__main__":
    _test()