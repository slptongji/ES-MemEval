from typing import Literal, Sequence
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from lib.shared.document_stores.document_store import DocumentStore
from lib.shared.prompt_strategies.prompt_strategy import PromptStrategy
from lib.shared.prompt_strategies.session_information import SessionInformation


class SessionWiseMemoryPrependStrategy(PromptStrategy):
    def __init__(self, store: DocumentStore) -> None:
        super().__init__()
        self.store: DocumentStore = store
    
    def _as_text(self, message: BaseMessage):
        role: Literal["Unknown", "System", "Ai", "Human"] = self._get_role_of(message)
        text: str = message.text()
        return f"{role}: {text}"

    def record_session(self,
                       session_history: Sequence[BaseMessage], 
                       session_information: SessionInformation):
        messages: list[str] = []

        message: BaseMessage
        for message in session_history:
            messages.append(self._as_text(message))
        
        message_indices: Sequence[str | None] | None = session_information["message_indices"]
        if message_indices is None:
            message_indices = []
        message_indices = [i for i in message_indices if i is not None]
        self.store.extend([Document(
            page_content=f"[{session_information["date"]}]\n{"\n".join(messages)}", 
            metadata={
                "message_indices": message_indices
            })])
    
    def generate_append_prompts(self, 
                                session_history: Sequence[BaseMessage], 
                                session_information: SessionInformation) -> Sequence[BaseMessage]:
        return []
    
    def generate_inplace_prompts(self, 
                                 session_history: Sequence[BaseMessage], 
                                 session_information: SessionInformation) -> str | None:
        return None

    def generate_prepend_prompts(self, 
                                 session_history: Sequence[BaseMessage], 
                                 session_information: SessionInformation) -> Sequence[BaseMessage]:
        if len(session_history) == 0:
            return []
        
        documents: Sequence[Document] = self.store.retrieve(str(session_history[-1].text()))
        if len(documents) == 0:
            return []
        
        result: list[BaseMessage] = []
        
        def sorter(document: Document) -> int:
            date: str = document.page_content.split("\n", 1)[0].strip("[]")
            try:
                year: str
                month: str
                day: str
                year, month, day = date.split("-", 2)
                return int(year) * 100 * 100 + int(month) * 100 + int(day)
            except:
                return 0

        document: Document
        for document in sorted(documents, key=sorter):
            lines: list[str] = document.page_content.split("\n")
            date: str = lines.pop(0).strip("[]")
            if date == "None":
                result.append(SystemMessage(f"The following dialogue happens on an unknown date."))
            else:
                result.append(SystemMessage(f"The following dialogue happens on {date}."))
            lines.append("System: An extra message to make the final message included.")

            message_type: str | None = None
            message: str = ""

            line: str
            for line in lines:
                if line.startswith(("System: ", "Ai: ", "Human: ")):
                    match message_type:
                        case "Ai": 
                            result.append(AIMessage(message))
                        case "Human": 
                            result.append(HumanMessage(message))
                        case None:
                            pass
                        case _: 
                            result.append(SystemMessage(message))
                    message_type, message = line.split(": ", 1)
                else:
                    message += f"\n{line}"

        result.append(SystemMessage(f"The following dialogue happens now."))
        return result
