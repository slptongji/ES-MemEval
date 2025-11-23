from typing import Literal, Sequence
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, SystemMessage

from lib.shared.document_stores.document_store import DocumentStore
from lib.shared.prompt_strategies.prompt_strategy import PromptStrategy
from lib.shared.prompt_strategies.session_information import SessionInformation


class TurnWiseMemoryInplaceStrategy(PromptStrategy):
    def __init__(self, store: DocumentStore) -> None:
        super().__init__()
        self.store: DocumentStore = store
    
    def _as_text(self, message: BaseMessage, 
                 session_information: SessionInformation):
        role: Literal["Unknown", "System", "Ai", "Human"] = self._get_role_of(message)
        text: str = message.text()
        if role == "Human" and session_information["human_name"] is not None:
            return f"{session_information["human_name"]}: {text}"
        if role == "Ai" and session_information["ai_name"] is not None:
            return f"{session_information["ai_name"]}: {text}"
        return f"{role}: {text}"

    def record_session(self,
                       session_history: Sequence[BaseMessage], 
                       session_information: SessionInformation):
        message_indices: Sequence[str | None] | None = session_information["message_indices"]
        if message_indices is None:
            message_indices = [None for _ in session_history]

        documents: list[Document] = []

        message: BaseMessage
        message_index: str | None
        for message, message_index in zip(session_history, message_indices):
            documents.append(Document(
                page_content=self._as_text(message, session_information),
                metadata={
                    "date": session_information["date"],
                    "message_indices": [] if message_index is None else [message_index]
                }))
        
        self.store.extend(documents)
        
    def generate_inplace_prompts(self, 
                                 session_history: Sequence[BaseMessage], 
                                 session_information: SessionInformation) -> str | None:
        if len(session_history) == 0:
            return None
        
        documents: Sequence[Document] = self.store.retrieve(str(session_history[-1].text()))
        if len(documents) == 0:
            return None
        
        result: str = ""
        index: int
        document: Document
        for index, document in enumerate(documents, start=1):
            result += f"{index}. [{document.metadata["date"]}] {document.page_content}\n"
        return result

    def generate_append_prompts(self, 
                                session_history: Sequence[BaseMessage], 
                                session_information: SessionInformation) -> Sequence[BaseMessage]:
        return []
    
    def generate_prepend_prompts(self, 
                                 session_history: Sequence[BaseMessage], 
                                 session_information: SessionInformation) -> Sequence[BaseMessage]:
        return []
    