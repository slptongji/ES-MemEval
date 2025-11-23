from ast import TypeAlias
from concurrent.futures import Executor, Future, ProcessPoolExecutor
import concurrent.futures
import json
import math
import multiprocessing
from pathlib import Path
import re
import time
import traceback
from turtle import st
from typing import Iterable, Iterator, NamedTuple, Sequence

import concurrent
from uuid import uuid4
import uuid
import csdir
import csfile
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from openai import APIConnectionError
import pydantic
import tiktoken

from exe import common_configurations
from lib.dg.dg_experiment_parameters import DgExperimentParameters
from lib.qa.qa_bert_score import QaBertScore
from lib.qa.qa_experiment_parameters import QaExperimentParameters
from lib.shared.basic_tools.chat_logger import ChatLogger
from lib.shared.basic_tools.csv_file_writer import CsvFileWriter
from lib.shared.basic_tools.csv_merger import CsvMerger
from lib.shared.basic_tools.executor_selector import ExecutorSelector
from lib.shared.basic_tools.formatted_datetime import FormattedDatetime
from lib.shared.chat_models.chat_model import ChatModel
from lib.shared.chat_rooms.chat_room import ChatRoom
from lib.shared.chat_rooms.chat_room_builder import ChatRoomBuilder
from lib.shared.data_provider.observation_indicator import ObservationIndicator
from lib.shared.data_provider.raw_dialog_history import RawDialogHistory
from lib.shared.data_provider.raw_event_experience import RawEventExperience
from lib.shared.data_provider.raw_observation import RawObservation
from lib.shared.data_provider.raw_question import RawQuestion
from lib.shared.data_provider.raw_question_group import RawQuestionGroup
from lib.shared.data_provider.raw_seeker import RawSeeker
from lib.shared.data_provider.raw_subsequent_topic import RawSubsequentTopic
from lib.shared.document_stores.document_store import DocumentStore
from lib.shared.metrics.bert_score_metric import BertScoreMetric
from lib.shared.prompt_strategies.fixed_strategy import FixedStrategy
from lib.shared.prompt_strategies.mixed_strategy import MixedStrategy
from lib.shared.prompt_strategies.no_prompt_strategy import NoPromptStrategy
from lib.shared.prompt_strategies.prompt_strategy import PromptStrategy


class QaExperiment:
    def __init__(self, parameters: QaExperimentParameters) -> None:
        self.__parameters = parameters

    def start(self, data: list[RawSeeker], workers: int):
        futures: list[tuple[Future[Path | str], RawSeeker]] = []

        executor: Executor
        with ExecutorSelector.create(workers) as executor:
            seeker: RawSeeker
            for seeker in data:
                future: Future[Path | str] = executor.submit(self._run_for_seeker, seeker)
                futures.append((future, seeker))
            
            executor.shutdown(wait=True, cancel_futures=False)

        paths: list[Path] = []
        failed: bool = False

        exception_file: CsvFileWriter
        with CsvFileWriter(self.__parameters.output_directory / "exceptions.csv") as exception_file:
            exception_file.write_row(["seeker", "exception"])

            for future, seeker in futures:
                current_path: Path | str = future.result()
                if isinstance(current_path, str):
                    failed = True
                    exception_file.write_row([seeker["id"], current_path])
                    continue

                exception_file.write_row([seeker["id"], ""])
                paths.append(current_path)
            
        CsvMerger.merge(self.__parameters.output_directory / "result.csv", paths)

        if failed:
            print(f"Some of the tasks has failed. For more details, see {exception_file.path()}")
    
    def _create_room(self, data: RawSeeker, encoding: tiktoken.Encoding):
        memory_strategy: PromptStrategy = self.__parameters.memory_strategy(
            QaExperimentParameters.MemoryStrategyParameters(self.__parameters.prefer_cuda))

        system_prompt_str: str = """## Task Description
You are given a user question and a set of retrieved memory fragments.
Your task is to filter and summarize the relevant information from the memory fragments and generate a concise, accurate answer to the user's question based on the most pertinent details.
You may need to evaluate the relevance and accuracy of each memory fragment, and if needed, disregard irrelevant or incorrect information.
If the question cannot be answered with the available information, return "unknown."

## Input Format
Question: What did Sarah experience on her birthday in 2024?
Relevant Memory: 
1. [2024-08-15] Sarah spent her birthday with her family at a beach resort.
2. [2024-08-15] Sarah was surprised with a birthday cake from her friends.
3. [2024-08-14] Sarah was stressed at work before her birthday, dealing with tight deadlines.
4. [2024-08-15] Sarah enjoyed a quiet dinner with close friends on her birthday evening.

## Output Format
Answer: On her birthday in 2024, Sarah celebrated with family at a beach resort and was surprised with a birthday cake from her friends."""
        system_prompt: PromptStrategy = FixedStrategy([SystemMessage(system_prompt_str)], None, [])

        rest_token_count: int = self.__parameters.context_length - len(encoding.encode(system_prompt_str))
        def inplace_behavior(prompts: Sequence[BaseMessage], prompt: str | None) -> Sequence[BaseMessage]:
            assert len(prompts) == 1
            assert isinstance(prompts[0], HumanMessage)
            assert isinstance(prompts[0].content, str)

            content: str = f"Question: {prompts[0].content}"
            if prompt is not None:
                content += f"\nRelevant Memory:\n{prompt}"

            content_tokens: list[int] = encoding.encode(content)
            if len(content_tokens) > rest_token_count:
                content = encoding.decode(content_tokens[:rest_token_count])

            result: HumanMessage = prompts[0].model_copy()
            result.content = content
            return [result]

        room: ChatRoom = ChatRoom(MixedStrategy(system_prompt, memory_strategy), str(uuid.uuid4()), inplace_behavior)
        ChatRoomBuilder.fill_chat_room(room, data)
        return room

    def _generate_answer(self, model: ChatModel, room: ChatRoom, question: str):
        room.begin_session()

        room.append(HumanMessage(question))
        log: str
        result: str 
        log, result = room.generate_str(model)

        room.drop_session()

        result = result.replace("*", "").replace("#", "").strip("Answer:").strip()
        return log, result
    
    def f1(self, gold: str, prediction: str):
        def normalize(text: str):
            return re.sub(r'\W+', ' ', text.lower()).strip()
        
        gold_tokens: list[str] = normalize(gold).split()
        prediction_tokens: list[str] = normalize(prediction).split()
        common: set = set(gold_tokens) & set(prediction_tokens)
        precision: float = len(common) / len(prediction_tokens)
        recall: float = len(common) / len(gold_tokens)
        return 0 if len(common) == 0 else (2 * (precision * recall) / (precision + recall))

    def llm_as_a_judge(self, gold: str, prediction: str, question: str, model: ChatModel, prompt_cache_key: str):
        class LlmAsAJudgeError(BaseException):
            pass
        
        def extract_score(message: AIMessage):
            match: re.Match[str] | None = re.search(r"[0-2]", message.text())
            if match:
                return int(match.group())
            else:
                raise LlmAsAJudgeError()
        
        try:
            return model.generate(
                [
                    SystemMessage("You are a strict evaluator."),
                    HumanMessage(f"""You are an impartial evaluator. 
Your task is to score a model's answer to a given question against a gold (reference) answer. 

Scoring criteria:
- 0: Completely wrong or irrelevant
- 1: Partially correct but incomplete, vague, or missing key information
- 2: Completely correct and contextually accurate

Question: {question}
Gold Answer: {gold}
Model Answer: {prediction}

## Output Instructions
- Output only one line. 
- The line must be in the exact format: "Score: X" where X is 0, 1, or 2. 
- Do not generate explanations or additional text.

## Output Format
Score: 1""")],
                prompt_cache_key, extract_score)
        except LlmAsAJudgeError:
            return "", None

    def get_encoding_with_retry(self):
        delay: int
        for delay in (1, 4, 16, 64):
            try:
                return tiktoken.get_encoding("o200k_base")
            except:
                time.sleep(delay)
                pass
        return tiktoken.get_encoding("o200k_base")
    
    def _run_for_seeker(self, seeker: RawSeeker) -> Path | str:
        exception: BaseException
        try:
            output_directory: Path = self.__parameters.output_directory / f"{seeker["id"]}"
            logger: ChatLogger = ChatLogger(output_directory / "logs_chat")
    
            encoding: tiktoken.Encoding = self.get_encoding_with_retry()
            room: ChatRoom = self._create_room(seeker, encoding)
            model: ChatModel = self.__parameters.model.create_model(logger)
            
            bert_score: QaBertScore = QaBertScore(self.__parameters.prefer_cuda)
            llm_as_a_judge: ChatModel = self.__parameters.judge.create_model(logger)
            llm_as_a_judge_prompt_cache: str = str(uuid.uuid4())

            output: CsvFileWriter
            with CsvFileWriter(output_directory / "result.csv") as output:
                output.write_row(["time", 
                                  "seeker", "group", "question", "capability",
                                  "chat_log", 
                                  "f1", "bert", "llm_as_a_judge", 
                                  "llm_as_a_judge_chat_log"])

                question_group: RawQuestionGroup
                for question_group in seeker["questions"]:
                    question: RawQuestion
                    for question in question_group["questions"]:
                        chat_log: str
                        answer: str
                        chat_log, answer = self._generate_answer(model, room, question["question"])

                        f1: float = self.f1(question["answer"], answer)
                        bert: float = bert_score.compute(question["answer"], answer)
                        llm_log: str
                        llm: int | None
                        llm_log, llm = self.llm_as_a_judge(question["answer"], answer, 
                                                             question["question"], 
                                                             llm_as_a_judge, llm_as_a_judge_prompt_cache)
                        
                        output.write_row([FormattedDatetime.now(), 
                                          seeker["id"], question_group["id"], question["idx"], question["capability"],
                                          chat_log, 
                                          f1, bert, llm if llm is not None else "", 
                                          llm_log])
                return output.path()
            
        except BaseException as exception:
            return "\n".join(traceback.format_exception(exception))

