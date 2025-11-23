from pathlib import Path
from typing import Iterable
from uuid import uuid4

import csdir
from langchain_core.messages import SystemMessage
from exe import common_configurations
from lib.dg.dg_experiment import DgExperiment
from lib.dg.dg_experiment_parameters import DgExperimentParameters
from lib.qa.qa_experiment import QaExperiment
from lib.qa.qa_experiment_parameters import QaExperimentParameters
from lib.shared.basic_tools.formatted_datetime import FormattedDatetime
from lib.shared.basic_tools.output_path_builder import OutputPathBuilder
from lib.shared.chat_models.chat_model_indicator import ChatModelIndicator
from lib.shared.chat_rooms.chat_room import ChatRoom
from lib.shared.data_provider.raw_data_provider import RawDataProvider
from lib.shared.data_provider.raw_seeker import RawSeeker
from lib.shared.document_stores.always_all_document_store import AlwaysAllDocumentStore
from lib.shared.prompt_strategies.no_prompt_strategy import NoPromptStrategy
from lib.shared.prompt_strategies.prompt_strategy import PromptStrategy
from lib.shared.prompt_strategies.session_wise_memory_inplace_strategy import SessionWiseMemoryInplaceStrategy


def memory(parameters: QaExperimentParameters.MemoryStrategyParameters):
    store: AlwaysAllDocumentStore = AlwaysAllDocumentStore()
    strategy: PromptStrategy = SessionWiseMemoryInplaceStrategy(store, False)
    return strategy


def main():
    output_directory: Path = csdir.create_directory(Config.output_directory.absolute())

    experiment: QaExperiment = QaExperiment(QaExperimentParameters(
        output_directory, Config.model, 
        4096, Config.judge_model,
        memory, False))
    
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
        model: ChatModelIndicator = common_configurations.mistral8b.as_normal_vllm()
        judge_model: ChatModelIndicator = common_configurations.gpt4o
        multiprocessing_workers: int = common_configurations.multiprocessing_workers

    main()