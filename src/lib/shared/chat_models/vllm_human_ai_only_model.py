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


class VllmHumanAiOnlyModel(ChatModel):
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

    def _modify_prompt(self, prompt: ChatPromptValue):
        def get_role(message: BaseMessage) -> typing.Literal["SYSTEM", "AI", "HUMAN"]:
            if isinstance(message, SystemMessage):
                return "SYSTEM"
            if isinstance(message, AIMessage):
                return "AI"
            if isinstance(message, HumanMessage):
                return "HUMAN"
            raise ValueError(f"Unsupported message type: {type(message)}")

        def walk_ai(current_index: int, 
                    current_message: str) -> tuple[int, str]:
            while True:
                if current_index >= len(prompt.messages):
                    return current_index, current_message
                
                message: BaseMessage = prompt.messages[current_index]
                if get_role(message) != "AI":
                    return current_index, current_message
                
                if current_message == "":
                    current_message = message.text()
                else:
                    current_message += f"\n============\nAI:\n{message.text()}"
                current_index += 1

        def walk_human_system(current_index: int, 
                              current_message: str) -> tuple[int, str]:
            while True:
                if current_index >= len(prompt.messages):
                    return current_index, current_message
                
                message: BaseMessage = prompt.messages[current_index]
                role: typing.Literal["SYSTEM", "AI", "HUMAN"] = get_role(message)
                if role == "AI":
                    return current_index, current_message
                
                if current_message == "":
                    if role == "HUMAN":
                        current_message = message.text()
                    else:
                        current_message = f"{role}:\n{message.text()}"
                else:
                    current_message += f"\n============\n{role}:\n{message.text()}"
                current_index += 1

        result: list[BaseMessage] = []

        index: int
        message: str
        index, message = walk_ai(0, 
                                 "You are a large AI language model. Due to technical limitations, we can only pass messages in the format of Human Message - AI Message - Human Message - ... In the following process, system messages will be delivered using the content format 'SYSTEM: <message>', and '============' on a single line will be used to demarcate two consecutive human or AI messages.")
        result.append(SystemMessage(message))

        while index < len(prompt.messages):
            index, message = walk_human_system(index, "")
            result.append(HumanMessage(message))

            if index < len(prompt.messages):
                index, message = walk_ai(index, "")
                result.append(AIMessage(message))
        
        return ChatPromptValue(messages=result)

    def _generate(self, prompts: ChatPromptValue, prompt_cache_key: str | None) -> tuple[AIMessage, ChatPromptValue]:
        prompts = self._modify_prompt(prompts)
        result: BaseMessage = self.client.invoke(prompts)
        assert isinstance(result, AIMessage)
        return result, prompts
    
    def _generate_with_schema[T: BaseModel](self,    # pylint: disable=un-declared-variable
                                           prompts: ChatPromptValue, 
                                           schema: type[T], 
                                           prompt_cache_key: str | None) -> tuple[AIMessage, T, ChatPromptValue]:
        prompts = self._modify_prompt(prompts)

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
