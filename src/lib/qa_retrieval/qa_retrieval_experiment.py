from ast import TypeAlias
from concurrent.futures import Future, ProcessPoolExecutor
import concurrent.futures
import json
import math
import multiprocessing
from pathlib import Path
from random import Random
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
from langchain_huggingface import HuggingFaceEmbeddings
from openai import APIConnectionError
import pydantic
import tiktoken

from exe import common_configurations
from lib.dg.dg_experiment_parameters import DgExperimentParameters
from lib.qa.qa_bert_score import QaBertScore
from lib.qa.qa_experiment_parameters import QaExperimentParameters
from lib.qa_retrieval.qa_retrieval_experiment_parameters import QaRetrievalExperimentParameters
from lib.shared.basic_tools.chat_logger import ChatLogger
from lib.shared.basic_tools.csv_file_writer import CsvFileWriter
from lib.shared.basic_tools.csv_merger import CsvMerger
from lib.shared.basic_tools.formatted_datetime import FormattedDatetime
from lib.shared.basic_tools.safe_dict import SafeDict
from lib.shared.basic_tools.string_random import StringRandom
from lib.shared.chat_models.chat_model import ChatModel
from lib.shared.chat_models.fake_chat_model import FakeChatModel
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
from lib.shared.document_stores.always_all_document_store import AlwaysAllDocumentStore
from lib.shared.document_stores.document_store import DocumentStore
from lib.shared.document_stores.vector_document_store import VectorDocumentStore
from lib.shared.embedding_models.embedding_provider import EmbeddingProvider
from lib.shared.metrics.bert_score_metric import BertScoreMetric
from lib.shared.prompt_strategies.fixed_strategy import FixedStrategy
from lib.shared.prompt_strategies.mixed_strategy import MixedStrategy
from lib.shared.prompt_strategies.no_prompt_strategy import NoPromptStrategy
from lib.shared.prompt_strategies.prompt_strategy import PromptStrategy
from lib.shared.prompt_strategies.session_information import SessionInformation


class QaRetrievalExperiment:
    def __init__(self, parameters: QaRetrievalExperimentParameters) -> None:
        self.__parameters = parameters

    def start(self, data: list[RawSeeker], workers: int):
        stores: dict[str, AlwaysAllDocumentStore] = self._create_stores(data)
        random: Random = Random(self.__parameters.random_seed)

        futures: list[tuple[Future[Path | str], RawSeeker]] = []

        executor: ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers, 
                                 mp_context=multiprocessing.get_context("spawn")) as executor:
            seeker: RawSeeker
            for seeker in data:
                future: Future[Path | str] = executor.submit(self._run_for_seeker, 
                                                             seeker,
                                                             stores,
                                                             Random(random.randint(0, 1000)))
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
    
    def _create_stores(self, 
                       data: list[RawSeeker]):
        result: dict[str, AlwaysAllDocumentStore] = {}
        seeker: RawSeeker
        for seeker in data:
            store: AlwaysAllDocumentStore = AlwaysAllDocumentStore()
            strategy: PromptStrategy = self.__parameters.memory_strategy(
            QaRetrievalExperimentParameters.MemoryStrategyParameters(store))
            room: ChatRoom = ChatRoom(strategy)
            ChatRoomBuilder.fill_chat_room(room, seeker)
            result[seeker["id"]] = store

        random: StringRandom = StringRandom(Random(0), 10)
        store: AlwaysAllDocumentStore
        for store in result.values():
            document: Document
            for document in store.all_documents():
                SafeDict.insert(document.metadata, "qa_retrieval_document_id", random.next())

        return result
    
    def _build_candidates(self, question: RawQuestion, documents: Iterable[Document]):
        evidences: set[str] = set(question["evidence"])

        correct: list[Document] = []
        incorrect: list[Document] = []
        
        document: Document
        for document in documents:
            message_indices: set[str] = set(document.metadata["message_indices"])
            if len(evidences & message_indices) > 0:
                correct.append(document)
            else:
                incorrect.append(document)
        return correct, incorrect
    
    def _build_candidates_cross_user(self, 
                                     random: Random,
                                     correct: list[Document], 
                                     incorrect: list[Document],
                                     cross_user: list[Document]):
        result: list[Document] = []
        rest: int = self.__parameters.provided_candidates

        rest -= len(correct)
        assert rest > 0
        result.extend(correct)

        rest -= len(incorrect)
        if rest <= 0:
            result.extend(random.sample(incorrect, rest))
        else:
            result.extend(incorrect)
            result.extend(random.sample(cross_user, rest))
        
        random.shuffle(result)
        return result
    
    def recall(self, gold: set[str], prediction: list[str], k: int):
        if len(prediction) > k:
            prediction = prediction[:k]
        hit: set = set(prediction) & set(gold)
        return len(hit) / len(gold) if len(gold) != 0 else 0
    
    def ndcg(self, gold: set[str], prediction: list[str], k: int):
        if len(prediction) > k:
            prediction = prediction[:k]
        dcg: float = 0.0
        index: int
        retrieved: str
        for index, retrieved in enumerate(prediction, 2):
            if retrieved in gold:
                dcg += 1 / math.log2(index + 2)
        idcg: float = sum(1 / math.log2(i) for i in range(2, min(len(gold), k) + 2))
        return dcg / idcg if idcg > 0 else 0
    
    def _run_for_seeker(self, 
                        seeker: RawSeeker, 
                        stores: dict[str, AlwaysAllDocumentStore], 
                        random: Random) -> Path | str:
        exception: BaseException
        try:
            output_directory: Path = self.__parameters.output_directory / f"{seeker["id"]}"
            logger: ChatLogger = ChatLogger(output_directory / "logs_chat")

            output: CsvFileWriter
            with CsvFileWriter(output_directory / "result.csv") as output:
                candidate_count: list[int] = list(self.__parameters.retrieved_candidates)

                row: list[str | float] = ["time", "seeker", "group", "question", "capability"]
                for candidate in candidate_count:
                    row.append(f"recall_{candidate}")
                    row.append(f"ndcg_{candidate}")
                output.write_row(row)

                cross_user: list[Document] = []
                for key in stores:
                    if key != seeker["id"]:
                        cross_user.extend(stores[key].all_documents())
                
                embedding: HuggingFaceEmbeddings = EmbeddingProvider.bge_m3(self.__parameters.prefer_cuda)

                question_group: RawQuestionGroup
                for question_group in seeker["questions"]:
                    question: RawQuestion
                    for question in question_group["questions"]:
                        correct: list[Document]
                        incorrect: list[Document]
                        correct, incorrect = self._build_candidates(question, stores[seeker["id"]].all_documents())

                        candidates: list[Document] = self._build_candidates_cross_user(
                            random, correct, incorrect, cross_user)
                        
                        store: DocumentStore = VectorDocumentStore(max(candidate_count), 
                                                                   "FAISS", 
                                                                   embedding)
                        store.extend(candidates)

                        strategy: PromptStrategy = self.__parameters.memory_strategy(
                            QaRetrievalExperimentParameters.MemoryStrategyParameters(store))
                        room: ChatRoom = ChatRoom(strategy, inplace_behavior=lambda x, _: x)

                        room.begin_session()
                        room.append(HumanMessage(question["question"]))
                        room.generate(FakeChatModel(logger))

                        retrieved: Sequence[Document] = store.last_retrieved()

                        gold: set[str] = {d.metadata["qa_retrieval_document_id"] for d in correct}
                        prediction: list[str] = [d.metadata["qa_retrieval_document_id"] for d in retrieved]

                        row = [FormattedDatetime.now(), seeker["id"], 
                               question_group["id"], question["idx"], question["capability"]]
                        
                        for candidate in candidate_count:
                            row.append(self.recall(gold, prediction, candidate))
                            row.append(self.ndcg(gold, prediction, candidate))
                            
                        output.write_row(row)
                return output.path()
            
        except BaseException as exception:
            return "\n".join(traceback.format_exception(exception))

