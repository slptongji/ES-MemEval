from pathlib import Path
from typing import Iterable
from uuid import uuid4

import csdir
from langchain_core.messages import SystemMessage
from exe import common_configurations
from lib.dg.dg_experiment import DgExperiment
from lib.dg.dg_experiment_parameters import DgExperimentParameters
from lib.shared.basic_tools.formatted_datetime import FormattedDatetime
from lib.shared.basic_tools.output_path_builder import OutputPathBuilder
from lib.shared.chat_models.chat_model_indicator import ChatModelIndicator
from lib.shared.chat_rooms.chat_room import ChatRoom
from lib.shared.data_provider.raw_data_provider import RawDataProvider
from lib.shared.data_provider.raw_seeker import RawSeeker
from lib.shared.prompt_strategies.no_prompt_strategy import NoPromptStrategy


def room(parameters: DgExperimentParameters.SupporterChatRoomParameters) -> ChatRoom:
    room: ChatRoom = ChatRoom(NoPromptStrategy(), prompt_cache_key=str(uuid4()))
    room.begin_session()
    room.append(parameters.beginning_prompt)
    room.append(SystemMessage("The following dialogue happens now. (You have no memory, so don't make things up.)"))
    room.append(parameters.beginning_prompt)
    return room


def main():
    output_directory: Path = csdir.create_directory(Config.output_directory.absolute())

    experiment: DgExperiment = DgExperiment(DgExperimentParameters(
        output_directory, Config.seeker_model, 
        Config.supporter_model, Config.turn_evaluation_model,
        Config.turn_evaluation_model, Config.overall_evaluation_model,
        room, False, Config.overall_score_only))
    
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
        supporter_model: ChatModelIndicator = common_configurations.phi3
        seeker_model: ChatModelIndicator = common_configurations.gpt4o
        turn_evaluation_model: ChatModelIndicator = common_configurations.mistral24b
        overall_evaluation_model: ChatModelIndicator = common_configurations.gpt4o
        multiprocessing_workers: int = common_configurations.multiprocessing_workers
        overall_score_only: bool = False

    main()