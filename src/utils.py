"""Utility functions for the project."""

import json
import os
from typing import Iterable, Union


def escape(text: str) -> str:
    return json.dumps(text)


def unescape(text: str) -> str:
    output = json.loads(text)
    if not isinstance(output, str):
        raise ValueError("Invalid text")
    return output


def load_test_cases(
    path: Union[os.PathLike[str], str]
) -> Iterable[tuple[str, list[str]]]:
    with open(path, "r", encoding="utf-8") as dataset:
        yield from read_test_cases(dataset)


def read_test_cases(dataset: Iterable[str]) -> Iterable[tuple[str, list[str]]]:
    for line in dataset:
        entries = line.rstrip("\n").split("\t")
        assert entries

        pattern = unescape(entries[0])
        texts = [unescape(text) for text in entries[1:]]
        yield pattern, texts
