from typing import Literal, Sequence
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, SystemMessage

from lib.shared.document_stores.document_store import DocumentStore
from lib.shared.prompt_strategies.prompt_strategy import PromptStrategy
from lib.shared.prompt_strategies.session_information import SessionInformation


class SessionWiseMemoryInplaceStrategy(PromptStrategy):
    def __init__(self, store: DocumentStore, index: bool = True) -> None:
        super().__init__()
        self.index = index
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
        messages: list[str] = []

        message: BaseMessage
        for message in session_history:
            messages.append(self._as_text(message, session_information))
        
        message_indices: Sequence[str | None] | None = session_information["message_indices"]
        if message_indices is None:
            message_indices = []
        message_indices = [i for i in message_indices if i is not None]
        self.store.extend([Document(
            page_content=f"{"\n".join(messages)}", 
            metadata={
                "date": session_information["date"],
                "message_indices": message_indices
            })])
        
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
    