from curses import nonl
from typing import Literal, Sequence
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, SystemMessage

from lib.shared.document_stores.document_store import DocumentStore
from lib.shared.prompt_strategies.prompt_strategy import PromptStrategy
from lib.shared.prompt_strategies.session_information import SessionInformation


class RoundWiseMemoryInplaceStrategy(PromptStrategy):
    def __init__(self, store: DocumentStore, index: bool = True) -> None:
        super().__init__()
        self.index = index
        self.store: DocumentStore = store
    
    def _as_text(self, message: BaseMessage, 
                 session_information: SessionInformation) -> tuple[Literal["Unknown", "System", "Ai", "Human"], str]:
        role: Literal["Unknown", "System", "Ai", "Human"] = self._get_role_of(message)
        text: str = message.text()
        if role == "Human" and session_information["human_name"] is not None:
            return "Human", f"{session_information["human_name"]}: {text}"
        if role == "Ai" and session_information["ai_name"] is not None:
            return "Ai", f"{session_information["ai_name"]}: {text}"
        return role, f"{role}: {text}"

    def record_session(self,
                       session_history: Sequence[BaseMessage], 
                       session_information: SessionInformation):
        message_indices: Sequence[str | None] | None = session_information["message_indices"]
        if message_indices is None:
            message_indices = [None for _ in session_history]

        documents: list[Document] = []

        current_messages: list[str] = []
        current_indices: list[str] = []

        def add_document():
            nonlocal current_messages
            nonlocal current_indices

            documents.append(Document(
                page_content="\n".join(current_messages),
                metadata={
                    "date": session_information["date"],
                    "message_indices": current_indices
                }))
            current_messages = []
            current_indices = []

        def add_message(text: str, index: str | None):
            current_messages.append(text)
            if index is not None:
                current_indices.append(index)

        supporter_found: bool = False

        message: BaseMessage
        message_index: str | None
        for message, message_index in zip(session_history, message_indices):
            role: Literal["Unknown", "System", "Ai", "Human"]
            text: str
            role, text = self._as_text(message, session_information)

            if role == "Ai":
                supporter_found = (len(current_messages) != 0)
                add_message(text, message_index)
                continue

            if supporter_found:
                add_document()
            
            add_message(text, message_index)
        
        if len(current_messages) != 0:
            add_document()

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
            if self.index:
                result += f"{index}. [{document.metadata["date"]}]\n{document.page_content}\n"
            else:
                result += f"[{document.metadata["date"]}]\n{document.page_content}\n"
        return result

    def generate_append_prompts(self, 
                                session_history: Sequence[BaseMessage], 
                                session_information: SessionInformation) -> Sequence[BaseMessage]:
        return []
    
    def generate_prepend_prompts(self, 
                                 session_history: Sequence[BaseMessage], 
                                 session_information: SessionInformation) -> Sequence[BaseMessage]:
        return []
    