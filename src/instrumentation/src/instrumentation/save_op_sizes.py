"""Compute the sizes of MERGE and CLONE operations, and save them to .npy files."""

import argparse
import os

import numpy as np


def main(args):
    """Compute the sizes of MERGE and CLONE operations, and save them to .npy files."""
    merge_sizes = []
    clone_sizes = []
    for filename in os.listdir(args.oplog_dir):
        with open(
            f"{args.oplog_dir}/{filename}", "r", encoding="utf-8"
        ) as file:
            logs = file.readlines()
            for log in logs:
                log = log[:-1]  # Remove newline
                if log.startswith("MERGE"):
                    size1, density1, size2, density2 = map(
                        float, log.split("\t")[1:]
                    )
                    merge_sizes.append((size1, density1, size2, density2))
                elif log.startswith("CLONE"):
                    size, density = map(float, log.split("\t")[1:])
                    clone_sizes.append((size, density))
    merge_sizes = np.array(merge_sizes)
    clone_sizes = np.array(clone_sizes)
    np.save(args.output_merge_sizes, merge_sizes)
    np.save(args.output_clone_sizes, clone_sizes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the sizes of MERGE and CLONE operations, and save them to .npy files."
    )
    parser.add_argument(
        "--oplog-dir",
        type=str,
        required=True,
        help="Directory containing operation logs.",
    )
    parser.add_argument(
        "--output-merge-sizes",
        type=str,
        required=True,
        help="Output .npy file to save MERGE sizes.",
    )
    parser.add_argument(
        "--output-clone-sizes",
        type=str,
        required=True,
        help="Output .npy file to save CLONE sizes.",
    )
    main(parser.parse_args())
