import csv
from pathlib import Path
from typing import Iterable
import typing

import csdir

if typing.TYPE_CHECKING:
    import _typeshed

class CsvFileWriter:
    def __init__(self, 
                 path: Path, 
                 mode: '_typeshed.OpenTextModeWriting | _typeshed.OpenTextModeUpdating' = "w",
                 create_directory: bool = True) -> None:
        self.__path = path.absolute()
        if create_directory:
            csdir.create_directory(self.__path.parent)
        self.__file = open(self.__path, mode=mode, newline='', encoding='utf8')
        self.__writer = csv.writer(self.__file)

    def path(self):
        return self.__path

    def write_row(self, row: Iterable[str | float]):
        self.__writer.writerow(row)
        self.__file.flush()

    def close(self):
        self.__file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    