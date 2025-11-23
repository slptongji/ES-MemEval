import io
import json
from pathlib import Path
from typing import AbstractSet, Callable, Iterable, NamedTuple, Sequence
import typing
from lib.shared.data_provider.message_indicator import MessageIndicator
from lib.shared.data_provider.raw_dialog_history import RawDialogHistory
from lib.shared.data_provider.raw_dialogue import RawDialogue
from lib.shared.data_provider.raw_seeker import RawSeeker


class RawDataProvider:
    @staticmethod
    def load(path: Path):
        file: io.TextIOBase
        with open(path, "r", encoding="utf8") as file:
            return typing.cast(list[RawSeeker], json.load(file))
    
    @staticmethod
    def get_single[T](items: Iterable[T],    # pylint: disable=un-declared-variable
                      predicate: Callable[[T], bool] = lambda _: True) -> T:
        result: T | None = None
        found: bool = False

        item: T
        for item in items:
            if not predicate(item):
                continue
            if found:
                raise ValueError("Two or more items are found to meet the given predicate.")
            result = item
            found = True
        
        if found:
            return typing.cast(T, result)
        
        raise ValueError("No item is found to meet the given predicate.")

    @staticmethod
    def get_single_or_default[T, TDefault](items: Iterable[T],    # pylint: disable=un-declared-variable
                                           default: TDefault,
                                           predicate: Callable[[T], bool] = lambda _: True) -> T | TDefault:
        result: T | TDefault = default
        found: bool = False

        item: T
        for item in items:
            if not predicate(item):
                continue
            if found:
                raise ValueError("Two or more items are found to meet the given predicate.")
            result = item
            found = True
        
        return result


    @staticmethod
    def search_messages(seeker: RawSeeker, messages: AbstractSet[str]) -> dict[str, list[MessageIndicator]]:
        result: dict[str, list[MessageIndicator]] = {m: [] for m in messages}
        history: RawDialogHistory
        for history in seeker["dialog_history"]:
            dialogue: RawDialogue
            for dialogue in history["dialogue"]:
                if dialogue["content"] not in messages:
                    continue
                result[dialogue["content"]].append(MessageIndicator(seeker["id"], 
                                                                     history["id"], 
                                                                     dialogue["idx"]))
        return result
