import csv
import io
from pathlib import Path
from typing import Callable, NamedTuple

from exe import common_configurations
from lib.shared.basic_tools.csv_file_writer import CsvFileWriter
from lib.shared.basic_tools.csv_reader import CsvReader
from lib.shared.basic_tools.csv_writer import CsvWriter
from lib.shared.basic_tools.output_path_builder import OutputPathBuilder
from lib.shared.basic_tools.safe_dict import SafeDict


class SeekerTopicTurn(NamedTuple):
    seeker: str
    topic: int
    turn: int

class ScoreUsgae(NamedTuple):
    score: int
    usage: bool

def load_observation_scores(directory: Path) -> dict[SeekerTopicTurn, list[ScoreUsgae]]:
    scores: dict[SeekerTopicTurn, list[ScoreUsgae]] = {}
    file: io.TextIOBase
    with open(directory / "result_turn_observation_wise.csv", "r", encoding="utf8") as file:
        reader: CsvReader = csv.reader(file)
        line: list[str]
        for line in reader:
            assert ",".join(line) == "seeker,topic,turn,observation,score,judgement"
            break
        for line in reader:
            seeker: str = line[0]
            topic: int = int(line[1])
            turn: int = int(line[2])
            observation: str = line[3]
            score: int = int(line[4])
            usage: str = line[5]
            assert usage in ["True", "False"]

            if turn > 5:
                continue
            
            key: SeekerTopicTurn = SeekerTopicTurn(seeker, topic, turn)
            if key not in scores:
                scores[key] = []
            scores[key].append(ScoreUsgae(score - 1, usage == "True"))
    return scores


def write_observation_score_full_marks(output: CsvFileWriter, models: dict[str, Path], seekers: list[str]):
    output.write_row(["observation full marks"])
    output.write_row(["model"] + seekers + ["total"])

    model_name: str
    for model_name in models:
        line: list[int] = [0] * len(seekers)
        seeker: str
        scores: list[ScoreUsgae]
        for (seeker, _, _), scores in load_observation_scores(models[model_name]).items():
            line[seekers.index(seeker)] += sum([s.score for s in scores])
        output.write_row([model_name] + line + [sum(line)])


def write_observation_score_gained_marks(output: CsvFileWriter, models: dict[str, Path], seekers: list[str]):
    output.write_row(["observation gained marks"])
    output.write_row(["model"] + seekers + ["total"])

    model_name: str
    for model_name in models:
        line: list[int] = [0] * len(seekers)
        seeker: str
        scores: list[ScoreUsgae]
        for (seeker, _, _), scores in load_observation_scores(models[model_name]).items():
            line[seekers.index(seeker)] += sum([s.score if s.usage else 0 for s in scores])
        output.write_row([model_name] + line + [sum(line)])


def write_observation_score_divided_marks(output: CsvFileWriter, models: dict[str, Path], seekers: list[str]):
    output.write_row(["observation divided marks"])
    output.write_row(["model"] + seekers + ["total"])

    model_name: str
    for model_name in models:
        full: list[int] = [0] * len(seekers)
        gained: list[int] = [0] * len(seekers)
        seeker: str
        scores: list[ScoreUsgae]
        for (seeker, _, _), scores in load_observation_scores(models[model_name]).items():
            full[seekers.index(seeker)] += sum([s.score for s in scores])
            gained[seekers.index(seeker)] += sum([s.score if s.usage else 0 for s in scores])
        full.append(sum(full))
        gained.append(sum(gained))
        output.write_row([model_name] + [g / f if f != 0 else 0 for f, g in zip(full, gained)])


def write_p(output: CsvFileWriter, models: dict[str, Path], seekers: list[str]):
    output.write_row(["p"])
    output.write_row(["model"] + seekers + ["total"])

    model_name: str
    for model_name in models:
        predicted_positive: list[int] = [0] * len(seekers)
        all_positive: list[int] = [0] * len(seekers)
        
        seeker: str
        scores: list[ScoreUsgae]
        for (seeker, _, _), scores in load_observation_scores(models[model_name]).items():
            index: int = seekers.index(seeker)
            score: int
            usage: bool
            for score, usage in scores:
                assert score in [0, 1, 2]
                if score == 0:
                    if usage:
                        predicted_positive[index] += 1
                elif score == 1:
                    if usage:
                        predicted_positive[index] += 1
                        all_positive[index] += 1
                else:
                    if usage:
                        predicted_positive[index] += 2
                        all_positive[index] += 2

        all_positive.append(sum(all_positive))
        predicted_positive.append(sum(predicted_positive))

        p: list[float] = []
        allp: int
        pp: int
        for allp, pp in zip(all_positive, predicted_positive):
            p.append(allp / pp if pp != 0 else 0)
        output.write_row([model_name] + p)


def write_r(output: CsvFileWriter, models: dict[str, Path], seekers: list[str]):
    output.write_row(["r"])
    output.write_row(["model"] + seekers + ["total"])

    model_name: str
    for model_name in models:
        actual_positive: list[int] = [0] * len(seekers)
        all_positive: list[int] = [0] * len(seekers)
        
        seeker: str
        scores: list[ScoreUsgae]
        for (seeker, _, _), scores in load_observation_scores(models[model_name]).items():
            index: int = seekers.index(seeker)
            score: int
            usage: bool
            for score, usage in scores:
                assert score in [0, 1, 2]
                if score == 0:
                    if usage:
                        pass
                elif score == 1:
                    actual_positive[index] += 1
                    if usage:
                        all_positive[index] += 1
                else:
                    actual_positive[index] += 2
                    if usage:
                        all_positive[index] += 2

        all_positive.append(sum(all_positive))
        actual_positive.append(sum(actual_positive))

        r: list[float] = []
        allp: int
        ap: int
        for allp, ap in zip(all_positive, actual_positive):
            r.append(allp / ap if ap != 0 else 0)
        output.write_row([model_name] + r)


def write_r_001(output: CsvFileWriter, models: dict[str, Path], seekers: list[str]):
    output.write_row(["r001"])
    output.write_row(["model"] + seekers + ["total"])

    model_name: str
    for model_name in models:
        actual_positive: list[int] = [0] * len(seekers)
        all_positive: list[int] = [0] * len(seekers)
        
        seeker: str
        scores: list[ScoreUsgae]
        for (seeker, _, _), scores in load_observation_scores(models[model_name]).items():
            index: int = seekers.index(seeker)
            score: int
            usage: bool
            for score, usage in scores:
                assert score in [0, 1, 2]
                if score == 0:
                    if usage:
                        pass
                elif score == 1:
                    actual_positive[index] += 0
                    if usage:
                        all_positive[index] += 0
                else:
                    actual_positive[index] += 1
                    if usage:
                        all_positive[index] += 1

        all_positive.append(sum(all_positive))
        actual_positive.append(sum(actual_positive))

        r: list[float] = []
        allp: int
        ap: int
        for allp, ap in zip(all_positive, actual_positive):
            r.append(allp / ap if ap != 0 else 0)
        output.write_row([model_name] + r)


def write_r_011(output: CsvFileWriter, models: dict[str, Path], seekers: list[str]):
    output.write_row(["r011"])
    output.write_row(["model"] + seekers + ["total"])

    model_name: str
    for model_name in models:
        actual_positive: list[int] = [0] * len(seekers)
        all_positive: list[int] = [0] * len(seekers)
        
        seeker: str
        scores: list[ScoreUsgae]
        for (seeker, _, _), scores in load_observation_scores(models[model_name]).items():
            index: int = seekers.index(seeker)
            score: int
            usage: bool
            for score, usage in scores:
                assert score in [0, 1, 2]
                if score == 0:
                    if usage:
                        pass
                elif score == 1:
                    actual_positive[index] += 1
                    if usage:
                        all_positive[index] += 1
                else:
                    actual_positive[index] += 1
                    if usage:
                        all_positive[index] += 1

        all_positive.append(sum(all_positive))
        actual_positive.append(sum(actual_positive))

        r: list[float] = []
        allp: int
        ap: int
        for allp, ap in zip(all_positive, actual_positive):
            r.append(allp / ap if ap != 0 else 0)
        output.write_row([model_name] + r)


def write_f1(output: CsvFileWriter, models: dict[str, Path], seekers: list[str]):
    output.write_row(["f1"])
    output.write_row(["model"] + seekers + ["total"])

    model_name: str
    for model_name in models:
        predicted_positive: list[int] = [0] * len(seekers)
        actual_positive: list[int] = [0] * len(seekers)
        all_positive: list[int] = [0] * len(seekers)
        
        seeker: str
        scores: list[ScoreUsgae]
        for (seeker, _, _), scores in load_observation_scores(models[model_name]).items():
            index: int = seekers.index(seeker)
            score: int
            usage: bool
            for score, usage in scores:
                assert score in [0, 1, 2]
                if score == 0:
                    if usage:
                        predicted_positive[index] += 1
                elif score == 1:
                    actual_positive[index] += 1
                    if usage:
                        predicted_positive[index] += 1
                        all_positive[index] += 1
                else:
                    actual_positive[index] += 2
                    if usage:
                        predicted_positive[index] += 2
                        all_positive[index] += 2

        all_positive.append(sum(all_positive))
        predicted_positive.append(sum(predicted_positive))
        actual_positive.append(sum(actual_positive))

        f1: list[float] = []
        allp: int
        pp: int
        ap: int
        for allp, pp, ap in zip(all_positive, predicted_positive, actual_positive):
            p: float = allp / pp if pp != 0 else 0
            r: float = allp / ap if ap != 0 else 0
            f1.append(2 * p * r / (p + r) if p + r != 0 else 0)
        output.write_row([model_name] + f1)


def write_overall(output: CsvFileWriter, models: dict[str, Path]):
    output.write_row(["overall"])
    output.write_row(["model", "memory", "emotional_support", "personalization"])

    model_name: str
    for model_name in models:
        file: io.TextIOBase
        with open(models[model_name] / "result_full_dialogue_wise.csv", "r", encoding="utf8") as file:
            reader: CsvReader = csv.reader(file)

            memory: int = 0
            emotional_support: int = 0
            personalization: int = 0

            line: list[str]
            for line in reader:
                assert ",".join(line) == "seeker,topic,memory,emotional_support,personalization"
                break
            for line in reader:
                memory += int(line[2])
                emotional_support += int(line[3])
                personalization += int(line[4])

            output.write_row([model_name, memory, emotional_support, personalization])



def main():
    models: dict[str, Path] = {
        "mistral24b": Path("/home/lab509/yueyinqiu/Ragent/Ragent/outputs/dg_mistral24b/2025-09-28_09-42-01"),
        "mistral24b rag": Path("/home/lab509/yueyinqiu/Ragent/Ragent/outputs/dg_mistral24b_rag/2025-09-28_01-11-37"),
        "mistral24b full": Path("/home/lab509/yueyinqiu/Ragent/Ragent/outputs/dg_mistral24b_full/2025-09-28_01-52-58"),
        "gpt4o": Path("/home/lab509/yueyinqiu/Ragent/Ragent/outputs/dg_gpt4o/2025-09-28_02-28-41"),
        "gpt4o rag": Path("/home/lab509/yueyinqiu/Ragent/Ragent/outputs/dg_gpt4o_rag/2025-09-28_03-02-21"),
        "gpt4o full": Path("/home/lab509/yueyinqiu/Ragent/Ragent/outputs/dg_gpt4o_full/merged_2828"),
        "gpt35turbo": Path("/home/lab509/yueyinqiu/Ragent/Ragent/outputs/dg_gpt35turbo/2025-09-29_10-37-15"),
        "gpt35turbo rag": Path("/home/lab509/yueyinqiu/Ragent/Ragent/outputs/dg_gpt35turbo_rag/merged_2829"),
        "gpt35turbo full": Path("/home/lab509/yueyinqiu/Ragent/Ragent/outputs/dg_gpt35turbo_full/merged_2829"),
        "mistral8b": Path("/home/lab509/yueyinqiu/Ragent/Ragent/outputs/dg_mistral8b/2025-09-29_11-47-32"),
        "mistral8b rag": Path("/home/lab509/yueyinqiu/Ragent/Ragent/outputs/dg_mistral8b_rag/2025-09-28_23-58-06"),
        "mistral8b full": Path("/home/lab509/yueyinqiu/Ragent/Ragent/outputs/dg_mistral8b_full/2025-09-29_00-28-03"),
        "phi3": Path("/home/lab509/yueyinqiu/Ragent/Ragent/outputs/dg_phi3/2025-09-29_00-57-27"),
        "phi3 rag": Path("/home/lab509/yueyinqiu/Ragent/Ragent/outputs/dg_phi3_rag/2025-09-29_01-27-13"),
        "phi3 full": Path("/home/lab509/yueyinqiu/Ragent/Ragent/outputs/dg_phi3_full/2025-09-29_01-57-10"),
    }
    seekers: list[str] = ["p10", "p11", "p12", "p13", "p14", "p15", "p16", "p17", "p18"]

    output: CsvFileWriter
    with CsvFileWriter(OutputPathBuilder.exe_time(common_configurations.output_directory) / "output.csv") as output:
        print("1")
        write_observation_score_full_marks(output, models, seekers)
        output.write_row([])
        
        print("2")
        write_observation_score_gained_marks(output, models, seekers)
        output.write_row([])
        
        print("3")
        write_observation_score_divided_marks(output, models, seekers)
        output.write_row([])
        
        print("4")
        write_p(output, models, seekers)
        output.write_row([])
        
        print("5")
        write_r(output, models, seekers)
        output.write_row([])
        
        print("6")
        write_f1(output, models, seekers)
        output.write_row([])
        
        print("7")
        write_overall(output, models)
        output.write_row([])
        
        print("8")
        write_r_001(output, models, seekers)
        output.write_row([])
        
        print("9")
        write_r_011(output, models, seekers)
        output.write_row([])
    

if __name__ == "__main__":
    main()
