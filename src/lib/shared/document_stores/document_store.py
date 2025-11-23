import abc
from typing import Iterable, Sequence
from langchain_core.documents import Document


class DocumentStore(abc.ABC):
    def __init__(self, candidate_size: int) -> None:
        super().__init__()
        self.__candidate_size: int = candidate_size
        self.__last_retrieved: Sequence[Document] = []

    def candidate_size(self) -> int:
        return self.__candidate_size
    
    @abc.abstractmethod
    def extend(self, documents: Iterable[Document]) -> None:
        pass

    @abc.abstractmethod
    def _retrieve(self, request: str) -> Sequence[Document]:
        pass

    def retrieve(self, request: str) -> Sequence[Document]:
        retrieved: Sequence[Document] = self._retrieve(request)
        self.__last_retrieved = retrieved
        return list(retrieved)

    def last_retrieved(self) -> Sequence[Document]:
        return list(self.__last_retrieved)

    @abc.abstractmethod
    def all_documents(self) -> Sequence[Document]:
        pass
