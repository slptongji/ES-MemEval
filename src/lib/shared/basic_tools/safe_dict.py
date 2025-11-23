from typing import MutableMapping


class SafeDict[TKey, TValue]:    # pylint: disable=un-declared-variable
    @staticmethod
    def insert(dict: MutableMapping[TKey, TValue], key: TKey, value: TValue):
        if key in dict:
            raise ValueError(f"Key {key} already exists in the mapping.")
        dict[key] = value

    @staticmethod
    def update(dict: MutableMapping[TKey, TValue], key: TKey, value: TValue):
        if key not in dict:
            raise ValueError(f"Key {key} does not exist in the mapping.")
        dict[key] = value