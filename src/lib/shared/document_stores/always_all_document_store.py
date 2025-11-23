from typing import Iterable

from langchain_core.documents import Document
from lib.shared.document_stores.document_store import DocumentStore


class AlwaysAllDocumentStore(DocumentStore):
    def __init__(self) -> None:
        super().__init__(2147483647)
        self._documents = []

    def extend(self, documents: Iterable[Document]) -> None:
        self._documents.extend(documents)

    def _retrieve(self, request: str):
        return self._documents

    def all_documents(self):
        return self._documents