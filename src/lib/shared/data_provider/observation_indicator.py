from typing import NamedTuple


class ObservationIndicator(NamedTuple):
    seeker: str
    session: str
    observation: int

    def as_str_index(self):
        return f"{self.session}:{self.observation}"
    
    @staticmethod
    def from_str_index(index: str, seeker: str):
        observation_str: str
        session: str
        session, observation_str = index.split(":", 1)
        observation_int: int = int(observation_str)
        return ObservationIndicator(seeker, session, observation_int)