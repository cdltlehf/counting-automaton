import argparse
import os

import pandas as pd
from tqdm import tqdm
from .constants import OP_NAMES


def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


def count_ops(logs):
    counts = {op: 0 for op in OP_NAMES}
    for log in logs:
        log = log[:-1]  # Remove newline
        fields = log.split("\t")
        # First field is operation name
        assert binary_search(OP_NAMES, fields[0]) >= 0
        counts[fields[0]] += 1
    df = pd.DataFrame(counts, columns=OP_NAMES, index=[0])
    return df


def main(args):
    op_counts_per_run = pd.DataFrame(columns=OP_NAMES, index=[])
    for filename in tqdm(list(os.listdir(args.oplog_dir))):
        with open(
            f"{args.oplog_dir}/{filename}", "r", encoding="utf-8"
        ) as file:
            op_counts_per_run = pd.concat([op_counts_per_run, count_ops(file)])
    op_counts_per_run.to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count operations per run from operation logs."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file to save the operation counts per run.",
    )
    parser.add_argument(
        "--oplog-dir",
        type=str,
        required=True,
        help="Directory containing operation logs.",
    )

    main(parser.parse_args())
