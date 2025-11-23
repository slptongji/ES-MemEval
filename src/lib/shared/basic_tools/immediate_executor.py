from collections.abc import Callable
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import ParamSpec, TypeVar


class ImmediateExecutor(Executor):
    _P = ParamSpec('_P')
    _T = TypeVar('_T')

    def submit(self, fn: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs) -> Future[_T]:
        future = Future()
        try:
            result = fn(*args, **kwargs)
        except BaseException as exception:
            future.set_exception(exception)
        else:
            future.set_result(result)
        return future