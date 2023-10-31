import os
import autode.exceptions as ex

from typing import Optional, List
from functools import cached_property
from autode.log import logger


class CalculationOutput:
    def __init__(self, filename: Optional[str] = None):
        self._filename = filename

    @property
    def filename(self) -> Optional[str]:
        return self._filename

    @filename.setter
    def filename(self, value: str):
        self._filename = str(value)
        self.clear()

    @cached_property
    def file_lines(self) -> List[str]:
        """
        Output files lines. This may be slow for large files but should
        not become a bottleneck when running standard DFT/WF calculations,
        are cached so only read once

        -----------------------------------------------------------------------
        Returns:
            (list(str)): Lines from the output file

        Raises:
            (autode.exceptions.NoCalculationOutput): If the file doesn't exist
        """
        logger.info("Setting output file lines")

        if self.filename is None or not os.path.exists(self.filename):
            raise ex.NoCalculationOutput

        file = open(self.filename, "r", encoding="utf-8", errors="ignore")
        return file.readlines()

    @property
    def exists(self) -> bool:
        """Does the calculation output exist?"""
        return self.filename is not None and os.path.exists(self.filename)

    def clear(self) -> None:
        """Clear the cached file lines"""

        if "file_lines" in self.__dict__:
            del self.__dict__["file_lines"]

        return None

    def try_to_print_final_lines(self, n: int = 50) -> None:
        """
        Attempt to print the final n output lines, if the output exists

        -----------------------------------------------------------------------
        Arguments:
            n: Number of lines
        """

        if self.exists:
            print("".join(self.file_lines[-n:]))

        return None


class BlankCalculationOutput(CalculationOutput):
    @property
    def filename(self) -> Optional[str]:
        return None

    @filename.setter
    def filename(self, value: str):
        raise ValueError("Cannot set the filename of a blank output")

    @property
    def file_lines(self) -> List[str]:
        return []

    @property
    def exists(self) -> bool:
        return True
