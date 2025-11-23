from pathlib import Path
from typing import Iterable
from uuid import uuid4

import csdir
from exe import common_configurations
from lib.dg.dg_experiment import DgExperiment
from lib.dg.dg_experiment_parameters import DgExperimentParameters
from lib.shared.basic_tools.formatted_datetime import FormattedDatetime
from lib.shared.basic_tools.output_path_builder import OutputPathBuilder
from lib.shared.chat_models.chat_model_indicator import ChatModelIndicator
from lib.shared.chat_rooms.chat_room import ChatRoom
from lib.shared.chat_rooms.chat_room_builder import ChatRoomBuilder
from lib.shared.data_provider.raw_data_provider import RawDataProvider
from lib.shared.data_provider.raw_seeker import RawSeeker
from lib.shared.document_stores.document_store import DocumentStore
from lib.shared.document_stores.vector_document_store import VectorDocumentStore
from lib.shared.embedding_models.embedding_provider import EmbeddingProvider
from lib.shared.prompt_strategies.fixed_strategy import FixedStrategy
from lib.shared.prompt_strategies.mixed_strategy import MixedStrategy
from lib.shared.prompt_strategies.session_wise_memory_prepend_strategy import SessionWiseMemoryPrependStrategy


def room(parameters: DgExperimentParameters.SupporterChatRoomParameters) -> ChatRoom:
    begining_prompt: FixedStrategy = FixedStrategy([parameters.beginning_prompt], None, [])
    store: DocumentStore = VectorDocumentStore(4, "FAISS", EmbeddingProvider.bge_m3(parameters.prefer_cuda))
    memory_prompt: SessionWiseMemoryPrependStrategy = SessionWiseMemoryPrependStrategy(store)
    room: ChatRoom = ChatRoom(MixedStrategy(begining_prompt, memory_prompt), prompt_cache_key=str(uuid4()))

    ChatRoomBuilder.fill_chat_room(room, parameters.data)
    room.begin_session()

    room.append(parameters.beginning_prompt)
    return room


def main():
    output_directory: Path = csdir.create_directory(Config.output_directory.absolute())

    experiment: DgExperiment = DgExperiment(DgExperimentParameters(
        output_directory, Config.seeker_model, 
        Config.supporter_model, Config.turn_evaluation_model,
        Config.turn_evaluation_model, Config.overall_evaluation_model,
        room, Config.prefer_cuda, Config.overall_score_only))
    
    all_seekers: list[RawSeeker] = RawDataProvider.load(Config.data_path)
    seekers: list[RawSeeker] = []
    skipped_seekers: set[str] = set(Config.skipped_seekers)
    i: int
    for i in range(len(all_seekers)):
        seeker: RawSeeker = all_seekers[i]
        if seeker["id"] in skipped_seekers:
            print(f"Seeker '{seeker["basic_info"]["name"]} ({seeker["id"]})' will be skipped "
                  f"according to the configuration.")
        else:
            seekers.append(seeker)
            
    print(f"The experiment will start soon. "
          f"To prevent errors from being overwritten by messages, the console will not output anything. "
          f"The outputs will be placed at {output_directory}")
    experiment.start(seekers, Config.multiprocessing_workers)
    print(f"Completed.")


if __name__ == "__main__":
    class Config:
        output_directory: Path = OutputPathBuilder.exe_time(common_configurations.output_directory)
        data_path: Path = common_configurations.data_path
        skipped_seekers: Iterable[str] = []
        supporter_model: ChatModelIndicator = common_configurations.mistral24b
        seeker_model: ChatModelIndicator = common_configurations.gpt4o
        turn_evaluation_model: ChatModelIndicator = common_configurations.mistral24b
        overall_evaluation_model: ChatModelIndicator = common_configurations.gpt4o
        multiprocessing_workers: int = common_configurations.multiprocessing_workers
        prefer_cuda: bool = common_configurations.prefer_cuda
        overall_score_only: bool = False

    main()