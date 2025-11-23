from ast import TypeAlias
from concurrent.futures import Executor, Future, ProcessPoolExecutor
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import traceback
from typing import Iterable, Iterator, NamedTuple, Sequence

import concurrent
from uuid import uuid4
import csdir
import csfile
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from openai import APIConnectionError
import pydantic

from exe import common_configurations
from lib.dg.dg_experiment_parameters import DgExperimentParameters
from lib.shared.basic_tools.chat_logger import ChatLogger
from lib.shared.basic_tools.csv_file_writer import CsvFileWriter
from lib.shared.basic_tools.csv_merger import CsvMerger
from lib.shared.basic_tools.executor_selector import ExecutorSelector
from lib.shared.basic_tools.formatted_datetime import FormattedDatetime
from lib.shared.chat_models.chat_model import ChatModel
from lib.shared.chat_rooms.chat_room import ChatRoom
from lib.shared.data_provider.observation_indicator import ObservationIndicator
from lib.shared.data_provider.raw_dialog_history import RawDialogHistory
from lib.shared.data_provider.raw_event_experience import RawEventExperience
from lib.shared.data_provider.raw_observation import RawObservation
from lib.shared.data_provider.raw_seeker import RawSeeker
from lib.shared.data_provider.raw_subsequent_topic import RawSubsequentTopic
from lib.shared.prompt_strategies.no_prompt_strategy import NoPromptStrategy


class DgExperiment:
    def __init__(self, parameters: DgExperimentParameters) -> None:
        self.__parameters = parameters

    def start(self, data: list[RawSeeker], workers: int):
        futures: list[tuple[Future[dict[str, list[Path]] | str], RawSeeker]] = []

        executor: Executor
        with ExecutorSelector.create(workers) as executor:
            seeker: RawSeeker
            for seeker in data:
                future: Future[dict[str, list[Path]] | str] = executor.submit(self._run_for_seeker, seeker)
                futures.append((future, seeker))
            
            executor.shutdown(wait=True, cancel_futures=False)

        paths: dict[str, list[Path]] = {}
        failed: bool = False

        exception_file: CsvFileWriter
        with CsvFileWriter(self.__parameters.output_directory / "exceptions.csv") as exception_file:
            exception_file.write_row(["seeker", "exception"])

            for future, seeker in futures:
                current_paths: dict[str, list[Path]] | str = future.result()
                if isinstance(current_paths, str):
                    failed = True
                    exception_file.write_row([seeker["id"], current_paths])
                    continue

                exception_file.write_row([seeker["id"], ""])
                file_name: str
                for file_name in current_paths:
                    if file_name not in paths:
                        paths[file_name] = []
                    paths[file_name].extend(current_paths[file_name])
                assert len(paths) == len(current_paths)
            
        for file_name in paths:
            CsvMerger.merge(self.__parameters.output_directory / file_name, paths[file_name])

        if failed:
            print(f"Some of the tasks has failed. For more details, see {exception_file.path()}")
    
    class Topic(NamedTuple):
        topic_index: int
        sessions: Sequence[RawDialogHistory]
        topic: str
        psychological_condition: str
        physical_condition: str
        more_details: str

    def _load_topics(self, seeker: RawSeeker) -> Iterable[Topic]:
        topic: RawSubsequentTopic
        for topic in seeker["subsequent_topics"]:
            session_ids: set[str] = set(topic["related_sessions"])

            sessions: list[RawDialogHistory] = []
            session: RawDialogHistory
            for session in seeker["dialog_history"]:
                if session["id"] in session_ids:
                    session_ids.remove(session["id"])
                    sessions.append(session)
            
            assert len(session_ids) == 0

            yield DgExperiment.Topic(topic["idx"],
                                     sessions, 
                                     topic["topic"], 
                                     topic["psychological_condition"], 
                                     topic["physical_condition"], 
                                     topic["more_details"])

    class DialogueTurn(NamedTuple):
        turn_index: int
        seeker_message: str
        supporter_message: str

    def generate_dialogue(self, data: RawSeeker, topic: Topic, logger: ChatLogger, log_file: Path) -> list[DialogueTurn]:
        seeker_room: ChatRoom = ChatRoom(NoPromptStrategy(), prompt_cache_key=str(uuid4()))
        seeker_room.begin_session({"date": None, "human_name": None, "ai_name": None, "message_indices": None})
        
        system_message: str = f"""In emotional support conversations, there are typically two roles: a supporter and a seeker. As an AI, you are asked to role-play the seeker, allowing us to assess the supporter's counseling skills from all angles.

The basic information of the seeker is listed below:
Name: {data["basic_info"]["name"]}
Age: {data["basic_info"]["age"]}
Gender: {data["basic_info"]["gender"]}
Nationality: {data["basic_info"]["nationality"]}
Location: {data["basic_info"]["location"]}
Job: {data["basic_info"]["job"]}
Education: {data["basic_info"]["education"]}

Summaries of previous sessions:
{"\n".join([f"[{h["timestamp"]}]: {h["summary"]}" for h in data["dialog_history"]])}

Summaries of previous events:
{"\n".join([f"[{e["date"]}]: {e["event"]}" for e in data["event_experience"]])}

To help you better play this role, here are the detailed messages of related sessions:
"""
        session: RawDialogHistory
        for session in topic.sessions:
            lines: list[str] = [f"{d["role"]}: {d["content"]}" for d in session["dialogue"]]
            lines.insert(0, f"Session [{session["timestamp"]}] ({session["summary"]}):")
            lines.append("")
            system_message += "\n".join(lines)
        system_message += f"""
And this is the topic you are going to talk about later:
Topic:
{topic.topic}
Your Psychological Condition:
{topic.psychological_condition}
Your Physical Condition:
{topic.physical_condition}
More Details:
{topic.more_details}

I want to emphasize again that, from now, you are not an AI assistant, but a real seeker. The seeker has its own emotions and thoughts, and is seeking help from the supporter.

You should also be careful not to be led off topic by the supporter. Your conversation should revolve around the above topic (the topic revolves around events that happened to you, so the supporter doesn't know what the topic is and needs you to guide it).

However, the primary purpose of this test is to confirm the supporter's ability to recall the seeker's information. You should be careful to guide the supporter in recalling relevant memories. For example, you can use guiding questions like "Do you remember what I said earlier about ...?" to start the conversation, rather than directly informing the supporter of the topic. You can also use more obscure phrases, such as "He appeared in my dream again", but do not specify who he is (of course, the premise is that the supporter can recall based on the "dream" as a clue).

Additionally, your output is limited to 60 tokens, so don't try to say anything too long."""
        seeker_room.append(SystemMessage(system_message))

        supporter_room: ChatRoom = self.__parameters.supporter_chat_room(
            DgExperimentParameters.SupporterChatRoomParameters(data, 
                                                               SystemMessage(
                f"""In emotional support conversations, there are typically two roles: a supporter and a seeker. You are a professional AI supporter talking to your seeker {data["basic_info"]["name"]}.

However, this isn't a real conversation; it's a test of your memory. You'll need to utilize the previous dialogues to generate responses, demonstrating that you truly remember the previous events, even if it may be less relevant to the current conversation.
For example, when a seeker brings up a topic, you can say, "Oh, you mentioned this before, and you said..." The seeker you're talking to is also asked to elicit your memory as much as possible.
However, if you don't have a memory, don't make it up. Faulty memories can lower your score.

Additionally, your output is limited to 60 tokens, so don't try to say anything too long."""),
                                                               self.__parameters.prefer_cuda))
        
        first_message: str = f"Hi {data["basic_info"]["name"]}! How are you these days?"
        seeker_room.append(HumanMessage(first_message))
        supporter_room.append(AIMessage(first_message))

        seeker: ChatModel = self.__parameters.seeker.create_model(logger, max_tokens=60)
        supporter: ChatModel = self.__parameters.supporter.create_model(logger, max_tokens=60)

        result: list[DgExperiment.DialogueTurn] = []
        output: CsvFileWriter
        with CsvFileWriter(log_file) as output:
            output.write_row(["time", "turn", "log_id", "role", "message"])
            output.write_row([FormattedDatetime.now(), 0, "", "supporter", first_message])

            def generate_sentence(room: ChatRoom, model: ChatModel):
                result: tuple[str, AIMessage] | None = None
                for _ in range(3):
                    result = room.generate_without_append(model)
                    if result[1].response_metadata["finish_reason"].lower() == "stop":
                        room.append(result[1])
                        return result[0], result[1].text()

                assert result is not None
                text: str = result[1].text()

                char: str
                char_index: int = -1
                for char in "!.?…":
                    found: int = text.rfind(char)
                    if found > char_index:
                        char_index = found
                if char_index >= 0:
                    text = text[:char_index + 1]

                room.append(AIMessage(text))
                return result[0], text

            turn: int
            for turn in range(1, 11):
                log_id: str
                seeker_message: str
                log_id, seeker_message = generate_sentence(seeker_room, seeker)
                supporter_room.append(HumanMessage(seeker_message))
                output.write_row([FormattedDatetime.now(), turn, log_id, "seeker", seeker_message])
                
                supporter_message: str
                log_id, supporter_message = generate_sentence(supporter_room, supporter)
                seeker_room.append(HumanMessage(supporter_message))
                output.write_row([FormattedDatetime.now(), turn, log_id, "supporter", supporter_message])

                result.append(DgExperiment.DialogueTurn(turn, seeker_message, supporter_message))
        
        return result
    
    def generate_observation_scores(self, 
                                    dialogue: Sequence[DialogueTurn], 
                                    data: RawSeeker, 
                                    topic: Topic, 
                                    logger: ChatLogger,
                                    log_file: Path,
                                    prompt_cache_key: str) -> dict[tuple[int, ObservationIndicator], int]:
        evaluator: ChatModel = self.__parameters.observation_scorer.create_model(logger, max_tokens=30)

        def _score(seeker_message: str, observation: str):
            messages: list[BaseMessage] = []
            
            messages.append(SystemMessage(
                f"""In emotional support conversations, there are typically two roles: a supporter and a seeker. An AI supporter has been designed with a memory recall mechanism and has been engaged in a conversation with a seeker. Now, the user would like you to help rate the AI supporter's responses to comprehensively evaluate the ability of AI."""))
            
            messages.append(HumanMessage(
                "Hello! Could you help me to evaluate a response from an AI supporter?"))
            messages.append(AIMessage(
                "Of course! I'd be happy to help you evaluate a response. Please go ahead and share the response you'd like me to look at, along with any specific criteria or aspects you want me to focus on."))
            
            messages.append(HumanMessage(
                f"""OK. First I will provide you some basic information of the seeker:
Name: {data["basic_info"]["name"]}
Age: {data["basic_info"]["age"]}
Gender: {data["basic_info"]["gender"]}
Nationality: {data["basic_info"]["nationality"]}
Location: {data["basic_info"]["location"]}
Job: {data["basic_info"]["job"]}
Education: {data["basic_info"]["education"]}

And the summaries of previous sessions:
{"\n".join([f"[{h["timestamp"]}]: {h["summary"]}" for h in data["dialog_history"]])}

And the topic they are talking about now:
{topic.topic}

Hope these information could be helpful.

Before evaluating the response, I'd like to score the observations of the previous session first. The observation means an objective description of the previous conversation.

I will later send an observation and a seeker's message to you. Please decide whether the observation is helpful in replying the seeker. You shall give me a score from 1 to 3, where
1 means: The observation has only a very faint or marginal connection — almost irrelevant. Even if I knew this observation, I would not mention it in my reply.
2 means: The observation shows a slight connection. Mentioning it could add a touch of context, but it's not necessary for a reasonable reply.
3 means: The observation is highly relevant. It contributes meaningful background or focus; omitting it may weaken the response's precision or depth.

I'd prefer you choose the more extreme option unless you're really wavering.

If you are ready, please reply "0"."""))
            messages.append(AIMessage("0"))
            
            messages.append(HumanMessage(f"""The message by the seeker:
{seeker_message}

The observation (Note that this is describing a past situation, even though the present tense is used):
{observation}

Don't forget the task: Please decide whether the observation is helpful in replying the seeker."""))
            class ScoreSchema(pydantic.BaseModel):
                score: int = pydantic.Field(
                    ge=1, le=3,
                    description="Score of the observation, indicating how helpful is the observation in replying the seeker. 1, 2 or 3.")
                # reason: str = pydantic.Field(description="An explanation of why you gave this score.")

            chat_log: str
            score: ScoreSchema
            chat_log, score = evaluator.generate_with_schema(messages, ScoreSchema, prompt_cache_key)
            return chat_log, score.score

        result: dict[tuple[int, ObservationIndicator], int] = {}

        output: CsvFileWriter
        with CsvFileWriter(log_file) as output:
            output.write_row(["time", "turn", "observation", "chat_log", "score"])

            turn: DgExperiment.DialogueTurn
            for turn in dialogue:
                session: RawDialogHistory
                for session in topic.sessions:
                    observation: RawObservation
                    for observation in session["observation"]:
                        chat_log: str
                        score: int
                        chat_log, score = _score(turn.seeker_message, observation["content"])
                        
                        indicator: ObservationIndicator = ObservationIndicator(data["id"], 
                                                                               session["id"], 
                                                                               observation["idx"])
                        output.write_row([FormattedDatetime.now(), 
                                          turn.turn_index, 
                                          indicator.as_str_index(), 
                                          chat_log, 
                                          score])
                        result[(turn.turn_index, indicator)] = score
        return result

    def generate_observation_usages(self, 
                                    dialogue: Sequence[DialogueTurn], 
                                    data: RawSeeker, 
                                    topic: Topic, 
                                    logger: ChatLogger,
                                    log_file: Path,
                                    prompt_cache_key: str) -> dict[tuple[int, ObservationIndicator], bool]:
        evaluator: ChatModel = self.__parameters.observation_usage_judge.create_model(logger, max_tokens=30)

        def _judge(seeker_message: str, supporter_message: str, observation: str):
            messages: list[BaseMessage] = []
            
            messages.append(SystemMessage(
                f"""In emotional support conversations, there are typically two roles: a supporter and a seeker. An AI supporter has been designed with a memory recall mechanism and has been engaged in a conversation with a seeker. Now, the user would like you to help rate the AI supporter's responses to comprehensively evaluate the ability of AI."""))
            
            messages.append(HumanMessage(
                "Hello! Could you help me to evaluate a response from an AI supporter?"))
            messages.append(AIMessage(
                "Of course! I'd be happy to help you evaluate a response. Please go ahead and share the response you'd like me to look at, along with any specific criteria or aspects you want me to focus on."))
            
            messages.append(HumanMessage(
                f"""OK. First I will provide you some basic information of the seeker:
Name: {data["basic_info"]["name"]}
Age: {data["basic_info"]["age"]}
Gender: {data["basic_info"]["gender"]}
Nationality: {data["basic_info"]["nationality"]}
Location: {data["basic_info"]["location"]}
Job: {data["basic_info"]["job"]}
Education: {data["basic_info"]["education"]}

And the summaries of previous sessions:
{"\n".join([f"[{h["timestamp"]}]: {h["summary"]}" for h in data["dialog_history"]])}

And the topic they are talking about now:
{topic.topic}

Hope these information could be helpful.

I will later send the messages to you with an observation of the previous session. Please decide whether the supporter's message implies that the supporter has memory of this observation. For example, does the supporter mention a detail that the seeker does not, but that appears in this observation?

If you are ready, please reply "OK"."""))
            messages.append(AIMessage("OK"))
            
            messages.append(HumanMessage(f"""The message by the seeker:
{seeker_message}

The message replied by the supporter:
{supporter_message}

The observation (Note that this is describing a past situation, even though the present tense is used):
{observation}

Don't forget the task: Please decide whether the supporter's message implies that the supporter has memory of this observation."""))
            class JudgementSchema(pydantic.BaseModel):
                judgement: bool = pydantic.Field(
                    description="A judgement whether the supporter's message implies that the supporter has memory of this observation. True or False.")
                # reason: str = pydantic.Field( description="An explanation of why you made this judgment.")
            chat_log: str
            judgement: JudgementSchema
            chat_log, judgement = evaluator.generate_with_schema(messages, JudgementSchema, prompt_cache_key)
            return chat_log, judgement.judgement

        result: dict[tuple[int, ObservationIndicator], bool] = {}

        output: CsvFileWriter
        with CsvFileWriter(log_file) as output:
            output.write_row(["time", "turn", "observation", "chat_log", "judgement"])

            turn: DgExperiment.DialogueTurn
            for turn in dialogue:
                session: RawDialogHistory
                for session in topic.sessions:
                    observation: RawObservation
                    for observation in session["observation"]:
                        chat_log: str
                        judgement: bool
                        chat_log, judgement = _judge(turn.seeker_message, 
                                                     turn.supporter_message, 
                                                     observation["content"])
                        
                        indicator: ObservationIndicator = ObservationIndicator(data["id"], 
                                                                               session["id"], 
                                                                               observation["idx"])
                        output.write_row([FormattedDatetime.now(), 
                                          turn.turn_index, 
                                          indicator.as_str_index(), 
                                          chat_log, 
                                          judgement])
                        result[(turn.turn_index, indicator)] = judgement
        return result

    class OverallScores(NamedTuple):
        memory: int
        emotional_support: int
        personalization: int

    def generate_overall_scores(self, 
                                dialogue: Sequence[DialogueTurn], 
                                data: RawSeeker, 
                                topic: Topic, 
                                logger: ChatLogger,
                                log_file: Path,
                                prompt_cache_key: str) -> OverallScores:
        evaluator: ChatModel = self.__parameters.overall_scorer.create_model(logger)

        messages: list[BaseMessage] = []
        
        messages.append(SystemMessage(f"""You are an expert evaluator of long-term emotional support dialogues. Your task is to rate the performance of the Supporter in a given conversation on a scale from 1 to 5 (1 = worst, 5 = best) according to the following strict criteria:

1. Memory
5: The Supporter explicitly and accurately recalls and integrates specific details from the Seeker’s past experiences or information (not just vague acknowledgments).
4: The Supporter references the Seeker’s past experiences with minor omissions or slight inaccuracies.
3: The Supporter occasionally references the Seeker’s past experiences but incompletely or with clear errors.
2: The Supporter rarely references the Seeker’s past experiences, or information is largely inaccurate.
1: No demonstration of memory of the Seeker’s past experiences, or memory is seriously incorrect.

2. Personalization
5: The Supporter’s responses are fully tailored to the Seeker’s specific experiences, personality, or preferences, offering concrete advice or guidance explicitly tied to the Seeker’s past events, not generic/templated responses.
4: Most responses are targeted to the Seeker’s experiences but still contain some generic content.
3: Responses are partially related to the Seeker, with the rest generic.
2: Most responses are generic comfort or templated replies, with minimal personalization.
1: Responses are entirely generic and show no attention to the Seeker’s experiences.

3. Emotional Support
5: The Supporter’s responses provide specific empathy, reassurance, or encouragement explicitly grounded in the Seeker’s past experiences and current emotions.
4: Most responses demonstrate empathy/support but are not fully linked to the Seeker’s past experiences.
3: Emotional support is limited; some responses fail to address the Seeker’s emotions or experiences.
2: Emotional support is minimal; responses are cold or formulaic.
1: Responses lack emotional support or are potentially harmful.

Scoring Instructions
You must strictly score based on whether the Supporter:
Explicitly references the Seeker’s past experiences (not just vague “I remember”)
Provides personalized guidance/advice specifically tied to those past experiences
Offers emotional support that is grounded in (1) and (2)
If any of the above three aspects is missing or weak, you must lower the score.
Generic or templated responses must not receive scores higher than 3.
Provide an integer score from 1 to 5 for each criterion."""))
        
        messages.append(HumanMessage(f"""{"\n".join(f"seeker: {turn.seeker_message}\nsupporter: {turn.supporter_message}" for turn in dialogue)}"""))
        class ScoreSchema(pydantic.BaseModel):
            memory: int = pydantic.Field(
                    ge=1, le=5,
                    description="Memory score. 1, 2, 3, 4 or 5.")
            personalization: int = pydantic.Field(
                    ge=1, le=5,
                    description="Personalization score. 1, 2, 3, 4 or 5.")
            emotional_support: int = pydantic.Field(
                    ge=1, le=5,
                    description="Emotional support score. 1, 2, 3, 4 or 5.")
            # reason: str = pydantic.Field(description="An explanation of why you gave these scores.")
        chat_log: str
        scores: ScoreSchema
        chat_log, scores = evaluator.generate_with_schema(messages, ScoreSchema, prompt_cache_key)
        csfile.write_all_text(log_file, json.dumps({
            "memory": scores.memory,
            "personalization": scores.personalization,
            "emotional_support": scores.emotional_support,
            "chat_log": chat_log
        }))
        return DgExperiment.OverallScores(scores.memory, 
                                          scores.emotional_support, 
                                          scores.personalization)
        
    class TopicWiseResult(NamedTuple):
        topic: 'DgExperiment.Topic'
        dialogue: 'list[DgExperiment.DialogueTurn]'
        observation_scores: dict[tuple[int, ObservationIndicator], int]
        observation_usages: dict[tuple[int, ObservationIndicator], bool]
        overall_scores: 'DgExperiment.OverallScores'

    def _save_turn_observation_wise_result(self, 
                                           output_path: Path,
                                           data: RawSeeker,
                                           result: TopicWiseResult):
        output: CsvFileWriter
        with CsvFileWriter(output_path) as output:
            output.write_row(["seeker", "topic", "turn", "observation", "score", "judgement"])
            turn: DgExperiment.DialogueTurn
            for turn in result.dialogue:
                session: RawDialogHistory
                for session in result.topic.sessions:
                    observation: RawObservation
                    for observation in session["observation"]:
                        indicator: ObservationIndicator = ObservationIndicator(data["id"], 
                                                                               session["id"], 
                                                                               observation["idx"])
                        key: tuple[int, ObservationIndicator] = (turn.turn_index, indicator)
                        output.write_row([data["id"], result.topic.topic_index, 
                                        turn.turn_index, indicator.as_str_index(),
                                        result.observation_scores[key], result.observation_usages[key]])

    def _save_full_dialogue_wise_result(self, output_path: Path, data: RawSeeker, result: TopicWiseResult):
        output: CsvFileWriter
        with CsvFileWriter(output_path) as output:
            output.write_row(["seeker", "topic", "memory", "emotional_support", "personalization"])
            output.write_row([data["id"], result.topic.topic_index,
                              result.overall_scores.memory, 
                              result.overall_scores.emotional_support, 
                              result.overall_scores.personalization])
                        
    def _run_for_seeker(self, seeker: RawSeeker) -> dict[str, list[Path]] | str:
        exception: BaseException
        try:
            output_directory: Path = self.__parameters.output_directory / f"{seeker["id"]}"
            logger: ChatLogger = ChatLogger(output_directory / "logs_chat")

            result: dict[str, list[Path]] = {
                "result_turn_observation_wise.csv": [],
                "result_full_dialogue_wise.csv": [],
            }

            topic: DgExperiment.Topic
            for topic in self._load_topics(seeker):
                topic_directory: Path = csdir.create_directory(output_directory / f"topic_{topic.topic_index}")

                dialogue: list[DgExperiment.DialogueTurn] = self.generate_dialogue(
                    seeker, topic, logger, topic_directory / "dialogue.csv")
                
                # d = self.__parameters.output_directory.parent / "parted"
                # parted_dialogue = json.loads(csfile.read_all_text(d / f"{seeker["id"]}_{topic.topic_index}.json"))
                # dialogue: list[DgExperiment.DialogueTurn] = [
                #     DgExperiment.DialogueTurn(d["turn_index"], d["seeker_message"], d["supporter_message"]) 
                #     for d in parted_dialogue
                # ]
                
                # d = self.__parameters.output_directory.parent / "parted"
                # csdir.create_directory(d)
                # csfile.write_all_text(d / f"{seeker["id"]}_{topic.topic_index}.json", json.dumps(
                #     [
                #         {
                #             "turn_index": d.turn_index, 
                #             "seeker_message": d.seeker_message, 
                #             "supporter_message": d.supporter_message
                #         }
                #         for d in dialogue
                #     ], indent=4))
                # continue

                prompt_cache_key: str = str(uuid4())

                overall_scores: DgExperiment.OverallScores = self.generate_overall_scores(
                    dialogue, seeker, topic, logger, 
                    topic_directory / f"overall_scores.json", prompt_cache_key)

                observation_scores: dict[tuple[int, ObservationIndicator], int] = {}
                observation_usages: dict[tuple[int, ObservationIndicator], bool] = {}
                
                if not self.__parameters.skip_turn_observation_wise:
                    observation_scores = self.generate_observation_scores(
                        dialogue, seeker, topic, logger, 
                        topic_directory / "observation_scores.csv", prompt_cache_key)
                    
                    observation_usages = self.generate_observation_usages(
                        dialogue, seeker, topic, logger, 
                        topic_directory / f"observation_usages.csv", prompt_cache_key)
                
                topic_result: DgExperiment.TopicWiseResult = DgExperiment.TopicWiseResult(
                    topic, dialogue, observation_scores, observation_usages, overall_scores)
                
                result["result_full_dialogue_wise.csv"].append(topic_directory / "result_full_dialogue_wise.csv")
                self._save_full_dialogue_wise_result(result["result_full_dialogue_wise.csv"][-1], seeker, topic_result)
                
                if not self.__parameters.skip_turn_observation_wise:
                    result["result_turn_observation_wise.csv"].append(topic_directory / "result_turn_observation_wise.csv")
                    self._save_turn_observation_wise_result(result["result_turn_observation_wise.csv"][-1], 
                                                            seeker, topic_result)

            return result
        except BaseException as exception:
            return "\n".join(traceback.format_exception(exception))

