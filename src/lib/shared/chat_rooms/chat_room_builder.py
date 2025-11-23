from pathlib import Path
from typing import Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from matplotlib.pyplot import hist

from lib.shared.chat_rooms.chat_room import ChatRoom
from lib.shared.data_provider.message_indicator import MessageIndicator
from lib.shared.data_provider.raw_data_provider import RawDataProvider
from lib.shared.data_provider.raw_dialog_history import RawDialogHistory
from lib.shared.data_provider.raw_dialogue import RawDialogue
from lib.shared.data_provider.raw_seeker import RawSeeker
from lib.shared.prompt_strategies.prompt_strategy import PromptStrategy
from lib.shared.prompt_strategies.session_information import SessionInformation


class ChatRoomBuilder:
    @staticmethod
    def fill_chat_room(room: ChatRoom, seeker: RawSeeker, ai_name: str = "supporter"):
        human_name: str = seeker["basic_info"]["name"]
        history: RawDialogHistory
        for history in seeker["dialog_history"]:
            room.begin_session({
                "date": history["timestamp"],
                "human_name": human_name,
                "ai_name": ai_name,
                "message_indices": [
                    MessageIndicator(seeker["id"], history["id"], d["idx"]).as_str_index()
                    for d in history["dialogue"]
                ]
            })
            ChatRoomBuilder.fill_session(room, history)
            room.end_session()

    @staticmethod
    def fill_session(room: ChatRoom, history: RawDialogHistory):
        dialogue: RawDialogue
        for dialogue in history["dialogue"]:
            if dialogue["role"] == "seeker":
                room.append(HumanMessage(dialogue["content"]))
            elif dialogue["role"] == "supporter":
                room.append(AIMessage(dialogue["content"]))
            else:
                raise ValueError()