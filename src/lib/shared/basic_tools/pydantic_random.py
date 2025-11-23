import random
from typing import Type
from unittest import mock

from pydantic import BaseModel
import pydantic_rng

class PydanticRandom:
    def __init__(self, random: random.Random):
        self.random = random

    def generate[T: BaseModel](self, type_: Type[T]):
        with (
            mock.patch("random.random", self.random.random),
            mock.patch("random.randint", self.random.randint),
            mock.patch("random.uniform", self.random.uniform),
            mock.patch("random.getrandbits", self.random.getrandbits),
            mock.patch("random.choices", self.random.choices)
        ):
            return pydantic_rng.generate(type_)