from pathlib import Path
from random import Random
from typing import Iterable, Sequence
import csdir
import csfile
import langchain_core
import langchain_core.load
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from lib.shared.basic_tools.string_random import StringRandom
from lib.shared.prompt_strategies.prompt_strategy import PromptStrategy

class ChatLogger:
    def __init__(self, directory: Path, random_seed: int | str | None = None) -> None:
        self.directory = csdir.create_directory(directory)
        self.random = StringRandom(Random(random_seed), 10, False)
    
    def log(self, messages: Sequence[BaseMessage]) -> str:
        id: str = self.random.next()
        directory: Path = csdir.create_directory(self.directory / id[0:2])
        csfile.write_all_text(directory / f"{id}.json", 
                              langchain_core.load.dumps(messages, pretty=True))
        csfile.write_all_lines(directory / f"{id}.txt", (f"{type(m)}:\n{m.text()}\n\n" for m in messages))
        return id


def _test():
    logger: ChatLogger = ChatLogger(Path("./outputs/test/chat_logger"), "0")
    print(logger.log([
        SystemMessage("This is a system message"),
        HumanMessage("Human message")
    ]))


if __name__ == "__main__":
    _test()