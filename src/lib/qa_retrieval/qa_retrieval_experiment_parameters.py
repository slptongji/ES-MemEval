from pathlib import Path
from typing import Callable, Iterable, NamedTuple

from lib.shared.document_stores.document_store import DocumentStore
from lib.shared.prompt_strategies.prompt_strategy import PromptStrategy


class QaRetrievalExperimentParameters(NamedTuple):
    output_directory: Path

    class MemoryStrategyParameters(NamedTuple):
        store: DocumentStore

    memory_strategy: Callable[[MemoryStrategyParameters], PromptStrategy]

    provided_candidates: int
    retrieved_candidates: Iterable[int]

    prefer_cuda: bool
    random_seed: str