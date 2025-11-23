from collections.abc import Callable
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from typing import ParamSpec, TypeVar

from lib.shared.basic_tools.immediate_executor import ImmediateExecutor


class ExecutorSelector:
    @staticmethod
    def create(workers: int) -> Executor:
        if workers == 0:
            return ImmediateExecutor()
        return ProcessPoolExecutor(max_workers=workers, 
                                   mp_context=multiprocessing.get_context("spawn"))