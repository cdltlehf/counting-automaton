"""Utility functions for the project."""

import json
import os
from typing import Iterable, Union

import numpy as np
import numpy.typing as npt


def get_outlier_bounds(data: npt.NDArray[np.float32]) -> tuple[float, float]:
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound


def escape(text: str) -> str:
    return json.dumps(text, indent=2)


def unescape(text: str) -> str:
    output = json.loads(text)
    if not isinstance(output, str):
        raise ValueError("Invalid text")
    return output


def load_test_cases(
    path: Union[os.PathLike[str], str],
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
