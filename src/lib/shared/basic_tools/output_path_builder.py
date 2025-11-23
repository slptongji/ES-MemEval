import inspect
import os
from pathlib import Path

import csdir
from lib.shared.basic_tools.formatted_datetime import FormattedDatetime


class OutputPathBuilder:
    @staticmethod
    def arg0_time(output_directory: Path, arg0: str, create: bool = True):
        import __main__
        output_directory = output_directory / arg0 / FormattedDatetime.now()
        output_directory = output_directory.absolute()
        if create:
            _ = csdir.create_directory(output_directory)
        return output_directory
    
    @staticmethod
    def exe_time(output_directory: Path, create: bool = True):
        import __main__
        output_directory = output_directory / Path(__main__.__file__).stem / FormattedDatetime.now()
        output_directory = output_directory.absolute()
        if create:
            _ = csdir.create_directory(output_directory)
        return output_directory
    
    @staticmethod
    def test_exe_time(output_directory: Path, create: bool = True):
        import __main__
        output_directory = output_directory / "test" / Path(__main__.__file__).stem / FormattedDatetime.now()
        output_directory = output_directory.absolute()
        if create:
            _ = csdir.create_directory(output_directory)
        return output_directory
    