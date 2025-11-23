from pathlib import Path
from typing import Iterable

import csdir
from exe import common_configurations
from lib.qa_retrieval.qa_retrieval_experiment import QaRetrievalExperiment
from lib.qa_retrieval.qa_retrieval_experiment_parameters import QaRetrievalExperimentParameters
from lib.shared.basic_tools.output_path_builder import OutputPathBuilder
from lib.shared.data_provider.raw_data_provider import RawDataProvider
from lib.shared.data_provider.raw_seeker import RawSeeker
from lib.shared.prompt_strategies.session_wise_memory_inplace_strategy import SessionWiseMemoryInplaceStrategy


def memory(parameters: QaRetrievalExperimentParameters.MemoryStrategyParameters):
    return SessionWiseMemoryInplaceStrategy(parameters.store, True)


def main():
    output_directory: Path = csdir.create_directory(Config.output_directory.absolute())

    experiment: QaRetrievalExperiment = QaRetrievalExperiment(QaRetrievalExperimentParameters(
        output_directory, memory,
        50, [2, 4, 6],
        Config.prefer_cuda, Config.random_seed))
    
    seekers: list[RawSeeker] = RawDataProvider.load(Config.data_path)
    
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
        multiprocessing_workers: int = common_configurations.multiprocessing_workers
        prefer_cuda: bool = common_configurations.prefer_cuda
        random_seed: str = "A3D5A09657BF4415AD10AEF72D7E9F3F"

    main()