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
from typing import Any, Iterable, Iterator, Literal, NamedTuple, Sequence

import concurrent
from uuid import uuid4
import uuid
import csdir
import csfile
import json_repair
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from openai import APIConnectionError
import pydantic
import rouge_score
import rouge_score.rouge_scorer
import rouge_score.scoring
import tiktoken

from exe import common_configurations
from lib.dg.dg_experiment_parameters import DgExperimentParameters
from lib.qa.qa_bert_score import QaBertScore
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
from lib.shared.data_provider.raw_summary import RawSummary
from lib.shared.document_stores.document_store import DocumentStore
from lib.shared.metrics.bert_score_metric import BertScoreMetric
from lib.shared.prompt_strategies.fixed_strategy import FixedStrategy
from lib.shared.prompt_strategies.mixed_strategy import MixedStrategy
from lib.shared.prompt_strategies.no_prompt_strategy import NoPromptStrategy
from lib.shared.prompt_strategies.prompt_strategy import PromptStrategy
from lib.sum.sum_experiment_parameters import SumExperimentParameters


class SumExperiment:
    def __init__(self, parameters: SumExperimentParameters) -> None:
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
            SumExperimentParameters.MemoryStrategyParameters(self.__parameters.prefer_cuda))

        system_prompt_str: str = """## Task Description
You are given a user question and a set of retrieved memory fragments 
(from long-term emotional support dialogues). 
Your task is to generate a concise, accurate summary answer to the question, 
integrating all relevant information across the fragments.

## Instructions
- Carefully read all memory fragments.  
- Identify which fragments are relevant to answering the question.  
- Summarize across multiple fragments, integrating events, emotions, and causal links.  
- Ensure the answer is coherent, factually grounded, and free of hallucinations.  
- If multiple fragments describe related events, merge them into a single concise narrative.  
- If the question cannot be answered from the available fragments, return `"unknown"`.  
- Keep the answer short and focused (3–5 sentences).  

## Input Format
Question: How did Nick cope with academic challenges over time?  
Relevant Memory:  
1. [2024-11-15] Nick lost interest in physics due to poor teaching quality.  
2. [2025-01-20] Nick started skipping classes and failing his courses.  
3. [2025-05-30] Nick was overwhelmed by assignments and considered dropping a course.  
4. [2025-07-05] Nick decided to switch majors after guidance from his professor.  
5. [2025-09-12] Nick’s workload became more manageable after adjusting to his new major.  

## Output Format
Answer: Nick initially lost interest in physics due to poor teaching and began skipping classes, which led to academic failures. By mid-2025, he was overwhelmed by assignments and considered dropping a course. After receiving guidance from his professor, he switched majors, which helped him manage his workload more effectively and regain academic stability."""
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
    
    class LlmJudgement(NamedTuple):
        chat_log: str
        score: int | str

        events_precision: float | str
        events_recall: float | str
        events_f1: float | str

    def llm_as_a_judge(self, gold: str, prediction: str, question: str, model: ChatModel, prompt_cache_key: str):
        class LlmAsAJudgeError(BaseException):
            pass
        
        class LlmResponse(NamedTuple):
            score: int
            events_reference: int
            events_generated: int
            events_recalled: int
        
        def extract_score(message: AIMessage) -> LlmResponse:
            try:
                response: Any = json_repair.loads(message.text())
                return LlmResponse(int(response["score"]), 
                                   int(response["num_events_reference"]), 
                                   int(response["num_events_generated"]), 
                                   int(response["num_events_recalled"]))
            except ValueError | json.JSONDecodeError:
                raise LlmAsAJudgeError()
        
        try:
            chat_log: str
            response: LlmResponse
            chat_log, response = model.generate(
                [
                    SystemMessage("""## Task Description
You are tasked with evaluating a model-generated summary against a human-written reference summary 
in the context of long-term emotional support dialogues. 
The evaluation should focus on:
1. Identifying distinct events in both summaries.
2. Determining how many reference events are correctly recalled in the generated summary.
3. Detecting hallucinations (events mentioned in the generated summary but not in the reference).
4. Assessing whether recalled events are consistent with the reference in temporal order, causal logic, and emotional meaning.

## Event Definition
- An **event** is a distinct state change, emotional shift, or decision explicitly mentioned in the summary. 
- Examples: "lost motivation due to poor teaching quality", "considered quitting a job", "reconciled with a friend".
- Explanations, adjectives, or background details are NOT counted as separate events unless they describe a clear action or state transition.

## Additional Requirements
- First extract **reference_events**: the list of distinct events from the reference summary.  
- Then extract **generated_events**: the list of distinct events from the generated summary.  
- Then identify **recalled_events**: the subset of generated_events that also appear in reference_events (semantic overlap allowed).  
- Use these lists to compute the following counts:
- **num_events_reference** = length of reference_events  
- **num_events_generated** = length of generated_events  
- **num_events_recalled** = length of recalled_events  
- Mark **hallucination = true** if generated_events contain unsupported events not present in reference_events; false otherwise.  

## Scoring Rubric (0–5 scale)
- 5 = Perfect: no hallucinations; all reference events recalled; consistent reasoning.  
- 4 = Strong: no hallucinations; most reference events (>75%) recalled; reasoning mostly consistent.  
- 3 = Fair: some omissions (~50% recalled) or minor inconsistencies; hallucinations absent or minimal.  
- 2 = Weak: major omissions (<50% recalled) or inconsistent reasoning; hallucinations present.  
- 1 = Poor: severe omissions (<25% recalled), multiple hallucinations, or strong inconsistencies.  
- 0 = Invalid: irrelevant or dominated by hallucinations.  

## Output Format (JSON)
{
  "score": <0–5>,
  "reference_events": [ ... ],
  "generated_events": [ ... ],
  "recalled_events": [ ... ],
  "num_events_reference": <int>,
  "num_events_generated": <int>,
  "num_events_recalled": <int>,
  "hallucination": true | false,
  "justification": "Brief explanation: which events were recalled, omitted, hallucinated, and how consistent the reasoning is."
}"""),
                    HumanMessage(f"""Question: {question}
Reference Summary: {gold}
Generated Summary: {prediction}""")],
                prompt_cache_key, extract_score)
            
            precision = response.events_recalled / response.events_generated if response.events_generated > 0 else 0
            recall = response.events_recalled / response.events_reference if response.events_reference > 0 else 0
            f1 = 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0
            return SumExperiment.LlmJudgement(chat_log, response.score, precision, recall, f1)
        except LlmAsAJudgeError:
            return SumExperiment.LlmJudgement("", "", "", "", "")

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
            
            rouge_scorer: rouge_score.rouge_scorer.RougeScorer = rouge_score.rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        
            llm_as_a_judge: ChatModel = self.__parameters.judge.create_model(logger)
            llm_as_a_judge_prompt_cache: str = str(uuid.uuid4())

            output: CsvFileWriter
            with CsvFileWriter(output_directory / "result.csv") as output:
                output.write_row(["time", 
                                  "seeker", "summary", "capability",
                                  "chat_log", 
                                  "rouge_1", "rouge_2", "rouge_L", 
                                  "event_precision", "event_recall", "event_f1",
                                  "llm_score", "llm_chat_log"])

                summary: RawSummary
                for summary in seeker["summaries"]:
                    chat_log: str
                    answer: str
                    chat_log, answer = self._generate_answer(model, room, summary["question"])

                    llm: SumExperiment.LlmJudgement = self.llm_as_a_judge(
                        summary["answer"], answer, summary["question"], 
                        llm_as_a_judge, llm_as_a_judge_prompt_cache)
                    
                    rouge: dict[Literal["rouge1", "rouge2", "rougeL"], rouge_score.scoring.Score] = rouge_scorer.score(
                        summary["answer"], answer)
                    
                    output.write_row([FormattedDatetime.now(), 
                                      seeker["id"], summary["idx"], summary["capability"],
                                      chat_log,
                                      rouge["rouge1"].fmeasure, rouge["rouge2"].fmeasure, rouge["rougeL"].fmeasure,
                                      llm.events_precision, llm.events_recall, llm.events_f1,
                                      llm.score, llm.chat_log])
                return output.path()
            
        except BaseException as exception:
            return "\n".join(traceback.format_exception(exception))

