"""Plot the range of counting operations."""

import argparse
import logging
from pathlib import Path
from typing import Any, TextIO, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import cai4py.parser_tools as pt
from cai4py.parser_tools.constants import MAX_REPEAT
from cai4py.parser_tools.constants import MAXREPEAT
from cai4py.parser_tools.constants import MIN_REPEAT
from cai4py.utils import get_outlier_bounds
from cai4py.utils import read_test_cases

Range = tuple[int, Union[int, float]]


def operand_to_range(operand: Any) -> Range:
    low, high = operand
    return low, (high if high is not MAXREPEAT else float("inf"))


def main(input_file: TextIO, output: Path) -> None:

    counting_ranges: list[Range] = []
    for pattern, _ in read_test_cases(input_file):
        parsed = pt.parse(pattern)
        counting_ranges += [
            operand_to_range(operand)
            for opcode, operand in pt.dfs(parsed)
            if opcode in {MAX_REPEAT, MIN_REPEAT}
        ]

    lows = np.array([low for low, _ in counting_ranges])
    highs = np.array([high for _, high in counting_ranges])

    high_inf = np.isinf(highs)
    # high_outlier_bounds = get_outlier_bounds(
    #     np.concat([highs[~high_inf], lows])
    # )
    high_outlier_bounds = (0, 512)
    low_outlier_bounds = high_outlier_bounds

    high_lower_bound, high_upper_bound = high_outlier_bounds
    low_lower_bound, low_upper_bound = low_outlier_bounds

    binstep = int(high_upper_bound // 8)
    tickstep = 2

    high_outlier = (
        (highs < high_lower_bound) | (high_upper_bound <= highs)
    ) & ~high_inf
    low_outlier = (lows < low_lower_bound) | (low_upper_bound <= lows)

    logging.info("High outlier_bounds: %s", high_outlier_bounds)
    logging.info("Low outlier_bounds: %s", low_outlier_bounds)
    logging.info("total: %d", len(counting_ranges))

    plt.figure(figsize=(2, 2))
    ax = plt.gca()

    low_inlier = ~low_outlier
    high_inlier = ~high_inf & ~high_outlier

    xs = lows[low_inlier & high_inlier]
    ys = highs[low_inlier & high_inlier].astype(int)
    # x_upper_line = np.max(xs) + 1
    # y_upper_line = np.max(ys) + 1
    x_upper_line = low_upper_bound
    y_upper_line = high_upper_bound

    xbins = list(range(0, x_upper_line, binstep)) + [
        x_upper_line + e * binstep for e in [0, 1]
    ]
    ybins = list(range(0, y_upper_line, binstep)) + [
        y_upper_line + e * binstep for e in [0, 1, 2]
    ]
    # Main plot
    norm = mpl.colors.LogNorm()
    bins = (xbins, ybins)
    _, _, _, image = ax.hist2d(xs, ys, bins=bins, cmap="Blues", norm=norm)
    logging.info("low_inlier, high_inlier: %d", len(xs))
    for xl, xh in zip([0, 128, 256, 384], [128, 256, 384, 512]):
        for yl, yh in zip([0, 128, 256, 384], [128, 256, 384, 512]):
            logging.info(
                "low_inlier [%d, %d), high_inlier [%d, %d): %d",
                xl,
                xh,
                yl,
                yh,
                len(xs[(xl <= xs) & (xs < xh) & (yl <= ys) & (ys < yh)]),
            )

    # Outlier wings
    xs = lows[low_inlier & high_outlier]
    ys = np.full_like(xs, ybins[-3])
    ax.hist2d(xs, ys, bins=bins, cmap="Blues", norm=norm)
    logging.info("low_inlier, high_outlier: %d", len(xs))

    ys = lows[low_outlier & high_inlier]
    xs = np.full_like(ys, xbins[-2])
    ax.hist2d(xs, ys, bins=bins, cmap="Blues", norm=norm)
    logging.info("low_outlier, high_inlier: %d", len(ys))

    xs = np.full_like(lows[low_outlier & high_outlier], xbins[-2])
    ys = np.full_like(xs, ybins[-3])
    ax.hist2d(xs, ys, bins=bins, cmap="Blues", norm=norm)
    logging.info("low_outlier, high_outlier: %d", len(xs))

    xs = lows[low_inlier & high_inf]
    ys = np.full_like(xs, ybins[-2])
    ax.hist2d(xs, ys, bins=bins, cmap="Blues", norm=norm)
    logging.info("low_inlier, high_inf: %d", len(xs))

    xs = lows[low_outlier & high_inf]
    ys = np.full_like(xs, ybins[-2])
    ax.hist2d(xs, ys, bins=bins, cmap="Blues", norm=norm)
    logging.info("low_outlier, high_inf: %d", len(xs))

    ax.set_aspect("equal")
    plt.colorbar(image, ax=ax)

    linewidth = mpl.rcParams["xtick.major.width"]
    ax.hlines(
        ybins[-2],
        xbins[0],
        xbins[-1],
        color="black",
        linestyles="--",
        linewidth=linewidth,
    )
    ax.hlines(
        ybins[-3],
        xbins[0],
        xbins[-1],
        color="black",
        linestyles="--",
        linewidth=linewidth,
    )
    ax.vlines(
        xbins[-2],
        ybins[0],
        ybins[-1],
        color="black",
        linestyles="--",
        linewidth=linewidth,
    )

    # ax.set_xticks([e for e in xbins[:-2:tickstep]] + list(xbins[-2:-1]))
    ax.set_xticks(xbins[::tickstep])
    ax.set_xticks(xbins[:-2], minor=True)
    ax.set_xticklabels([str(e) for e in xbins[::tickstep]])

    # ax.set_yticks([e for e in ybins[:-3:tickstep]] + list(ybins[-3:-2]))
    ax.set_yticks(ybins[:-2:tickstep] + [ybins[-2]])
    ax.set_yticks(ybins[:-2], minor=True)
    ax.set_yticklabels([str(e) for e in ybins[:-2:tickstep]] + [r"$\infty$"])

    ax.set_xlabel("Lower bound $l$")
    ax.set_ylabel("Upper bound $h$")
    # ax.set_title("Ranges of counting operations")

    plt.savefig(output, bbox_inches="tight", dpi=300)
    plt.savefig(output.with_suffix(".pgf"), bbox_inches="tight", dpi=300)
    plt.clf()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input", type=argparse.FileType("r"), help="Filtered patterns"
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.input, args.output)
