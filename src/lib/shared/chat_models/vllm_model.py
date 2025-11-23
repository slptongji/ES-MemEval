import typing
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_core.prompt_values import ChatPromptValue, PromptValue
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from pydantic import BaseModel, Field, SecretStr

from exe import common_configurations
from lib.shared.basic_tools.chat_logger import ChatLogger
from lib.shared.basic_tools.output_path_builder import OutputPathBuilder
from lib.shared.chat_models.chat_model import ChatModel


class VllmModel(ChatModel):
    def __init__(self, url: str, model: str, api_key: str, 
                 logger: ChatLogger, 
                 retry: typing.Sequence[float],
                 max_tokens: int) -> None:
        super().__init__(logger, retry)
        self.model_name = model
        self.client = ChatOpenAI(api_key=SecretStr(api_key),     # pyright: ignore[reportCallIssue]
                                 model=model,     # pyright: ignore[reportCallIssue]
                                 base_url=url,     # pyright: ignore[reportCallIssue]
                                 max_completion_tokens=max_tokens) # pyright: ignore[reportCallIssue]

    def _generate(self, prompts: ChatPromptValue, prompt_cache_key: str | None) -> tuple[AIMessage, ChatPromptValue]:
        result: BaseMessage = self.client.invoke(prompts)
        assert isinstance(result, AIMessage)
        return result, prompts
    
    def _generate_with_schema[T: BaseModel](self,    # pylint: disable=un-declared-variable
                                            prompts: ChatPromptValue, 
                                            schema: type[T], 
                                            prompt_cache_key: str | None) -> tuple[AIMessage, T, ChatPromptValue]:
        client: Runnable = self.client.with_structured_output(schema, include_raw=True, strict=True)

        class StructuredOutput(typing.TypedDict):
            raw: AIMessage
            parsed: T | None
            parsing_error: BaseException | None
        result: StructuredOutput = client.invoke(prompts)
        if result["parsing_error"] is not None:
            raise typing.cast(BaseException, result["parsing_error"])
        return result["raw"], typing.cast(T, result["parsed"]), prompts
    
    def estimate_token_count(self, message: str) -> int:
        raise NotImplementedError()

    def max_token_count(self) -> int:
        raise NotImplementedError()


def _test():
    logger: ChatLogger = ChatLogger(OutputPathBuilder.test_exe_time(common_configurations.output_directory))
    model: ChatModel = common_configurations.mistral24b.create_model(logger)
    print(model.generate([
        HumanMessage("Hello")
    ]))

    class WeatherFormatter(BaseModel):
        weather: str = Field(description="weather")
        wetness: int = Field(description="wetness")
        temperature: int = Field(description="temperature")
    print(model.generate_with_schema([
        HumanMessage("Hows the weather today?")
    ], schema=WeatherFormatter))

    model = common_configurations.mistral24b.create_model(logger, max_tokens=1000)
    print(model.generate([
        SystemMessage("Please reply the user with the longest response as you can."),
        HumanMessage("Hello!")
    ]))

    model = common_configurations.mistral24b.create_model(logger, max_tokens=100)
    print(model.generate([
        SystemMessage("Please reply the user with the longest response as you can."),
        HumanMessage("Hello!")
    ]))

    model = common_configurations.mistral24b.create_model(logger, max_tokens=20)
    print(model.generate([
        SystemMessage("Please reply the user with the longest response as you can."),
        HumanMessage("Hello!")
    ]))


if __name__ == "__main__":
    _test()