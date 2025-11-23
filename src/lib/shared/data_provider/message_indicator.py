from typing import NamedTuple


class MessageIndicator(NamedTuple):
    seeker: str
    session: str
    message: int

    def as_str_index(self):
        return f"{self.session}:{self.message}"
    
    @staticmethod
    def from_str_index(index: str, seeker: str):
        message_str: str
        session: str
        session, message_str = index.split(":", 1)
        message_int: int = int(message_str)
        return MessageIndicator(seeker, session, message_int)