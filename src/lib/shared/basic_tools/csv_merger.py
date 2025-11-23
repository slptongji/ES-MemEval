import io
from pathlib import Path
from typing import Iterator, Sequence
import csv

import csfile

from lib.shared.basic_tools.csv_reader import CsvReader
from lib.shared.basic_tools.csv_writer import CsvWriter


class CsvMerger:
    @staticmethod
    def merge(output: Path, inputs: Sequence[Path]):
        if len(inputs) == 0:
            csfile.write_all_text(output, "")
            return

        path_iterator: Iterator[Path] = iter(inputs)

        path: Path = next(path_iterator)
        file: io.TextIOBase
        with open(path, mode='r', newline='', encoding='utf8') as file:
            header: list[str] = next(csv.reader(file))
            header_set: set[str] = set(header)
            
        for path in path_iterator:
            with open(path, mode='r', newline='', encoding='utf8') as file:
                header = next(csv.reader(file))
                if header_set != set(header):
                    raise ValueError(f"Header mismatch in file {path}. Expected {header_set}, got {set(header)}.")
        
        output_file: io.TextIOBase
        with open(output, mode='w', newline='', encoding='utf8') as output_file:
            writer: CsvWriter = csv.writer(output_file)
            writer.writerow(header)

            for path in inputs:
                with open(path, mode='r', newline='', encoding='utf8') as file:
                    reader: CsvReader = csv.reader(file)
                    _ = next(reader)
                    row: list[str]
                    for row in reader:
                        writer.writerow(row)