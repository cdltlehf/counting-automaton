"""Analysis pattern."""

import argparse
import logging
import math
import os
from pathlib import Path
import sys
from typing import Any, TypedDict, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import parser_tools as pt
from parser_tools.constants import MAX_REPEAT
from parser_tools.constants import MAXREPEAT
from parser_tools.constants import MIN_REPEAT
from utils import get_outlier_bounds
from utils import unescape


class AnalysisResults(TypedDict):
    total: int
    semi_trivial: int
    trivial: int
    max_high: list[float]
    max_low: list[float]
    max_gap: list[float]
    max_sparse_set_size: list[float]


Range = tuple[int, Union[int, float]]


def is_trivial(counting_range: Range) -> bool:
    low, high = counting_range
    return low in {0, 1} and high in {1, MAXREPEAT}


def is_semi_trivial(counting_range: Range) -> bool:
    low, high = counting_range
    return low in {0, 1} or high in {1, MAXREPEAT}


def get_sparse_set_size(counting_range: Range) -> float:
    low, high = counting_range
    if high is MAXREPEAT:
        return 2
    assert isinstance(high, int)
    k = high - low
    return math.ceil(high / (k + 1)) * 2


def hist_with_inf(ax: mpl.axes.Axes, data_like: list[float]) -> None:
    data = np.array(data_like, dtype=float)
    lower_bound, upper_bound = get_outlier_bounds(data)

    lower_outliers = data[data < lower_bound]
    upper_outliers = data[(data > upper_bound) & np.isfinite(data)]
    infinite_outliers = data[~np.isfinite(data)]

    filtered_data = np.clip(data, lower_bound, upper_bound).astype(int)
    min_data = filtered_data.min()
    max_data = filtered_data.max()
    gap = math.ceil((max_data - min_data + 1) / 20)
    n = math.ceil((max_data - min_data) / gap)
    if gap == 0:
        bins = np.array(range(min_data, max_data + 1), dtype=int)
    else:
        bins = np.array(range(min_data, min_data + gap * n + 1, gap), dtype=int)
    bins[-1] = upper_bound

    ax.hist(
        filtered_data,
        bins=list(bins),
        edgecolor="tab:blue",
        facecolor="tab:blue",
    )
    bin_width = bins[1] - bins[0]
    ax.set_xticks(bins)
    ax.set_xticklabels(bins, ha="left")

    lower_x = bins[0] - bin_width
    upper_x = bins[-1] + bin_width
    infinite_x = bins[-1] + bin_width

    logging.debug("Bins: %s", bins)

    logging.debug(
        "Range of non-outliers: (%d, %d)",
        min(filtered_data),
        max(filtered_data),
    )

    xlim_left, xlim_right = ax.get_xlim()
    if len(lower_outliers) > 0:
        box = mpl.patches.Rectangle(
            (lower_x - bin_width, 0),
            bin_width,
            len(lower_outliers),
            edgecolor="tab:red",
            facecolor="tab:red",
        )
        ax.add_patch(box)
        xlim_left = lower_x - bin_width

    if len(upper_outliers) > 0:
        logging.debug(
            "Range of upper-outliers: (%d, %d)",
            min(upper_outliers),
            max(upper_outliers),
        )
        box = mpl.patches.Rectangle(
            (upper_x, 0),
            bin_width,
            len(upper_outliers),
            edgecolor="tab:red",
            facecolor="tab:red",
        )
        infinite_x += bin_width
        ax.add_patch(box)
        xlim_right = upper_x + bin_width

    if len(infinite_outliers) > 0:
        box = mpl.patches.Rectangle(
            (infinite_x, 0),
            bin_width,
            len(infinite_outliers),
            edgecolor="tab:green",
            facecolor="tab:green",
        )
        ax.add_patch(box)
        xlim_right = infinite_x + bin_width
    ax.set_xlim(xlim_left, xlim_right + 1)


def operand_to_range(operand: Any) -> Range:
    low, high = operand
    return low, (high if high is not MAXREPEAT else float("inf"))


def main(output_dir: Path) -> None:
    logging.basicConfig(level=logging.INFO)

    analysis_results = AnalysisResults(
        total=0,
        semi_trivial=0,
        trivial=0,
        max_high=[],
        max_low=[],
        max_gap=[],
        max_sparse_set_size=[],
    )

    counting_ranges: list[Range] = []
    for pattern in map(unescape, sys.stdin):
        parsed = pt.parse(pattern)
        counting_ranges += [
            operand_to_range(operand)
            for opcode, operand in pt.dfs(parsed)
            if opcode in {MAX_REPEAT, MIN_REPEAT}
        ]
    os.makedirs(output_dir, exist_ok=True)

    lows = np.array([low for low, _ in counting_ranges])
    highs = np.array([high for _, high in counting_ranges])

    # ax = plt.gca()
    # logging.debug(output_dir / "max_high.pdf")
    # hist_with_inf(ax, analysis_results["max_high"])
    # plt.savefig(output_dir / "max_high.pdf")
    # plt.clf()

    # ax = plt.gca()
    # logging.debug(output_dir / "max_low.pdf")
    # hist_with_inf(ax, analysis_results["max_low"])
    # plt.savefig(output_dir / "max_low.pdf")
    # plt.clf()

    # ax = plt.gca()
    # logging.debug(output_dir / "max_gap.pdf")
    # hist_with_inf(ax, analysis_results["max_gap"])
    # plt.savefig(output_dir / "max_gap.pdf")
    # plt.clf()

    # ax = plt.gca()
    # logging.debug(output_dir / "max_sparse_set_size.pdf")
    # hist_with_inf(ax, analysis_results["max_sparse_set_size"])
    # plt.savefig(output_dir / "max_sparse_set_size.pdf")
    # plt.clf()

    high_inf = np.isinf(highs)

    # high_outlier_bounds = get_outlier_bounds(highs)
    # low_outlier_bounds = get_outlier_bounds(lows)
    high_outlier_bounds = (-1, 64)
    low_outlier_bounds = high_outlier_bounds

    binstep = 4
    tickstep = 16

    high_lower_bound, high_upper_bound = high_outlier_bounds
    low_lower_bound, low_upper_bound = low_outlier_bounds

    high_outlier = (
        (highs <= high_lower_bound) | (highs >= high_upper_bound)
    ) & ~high_inf
    low_outlier = (lows <= low_lower_bound) | (lows >= low_upper_bound)

    logging.info("High outlier_bounds: %s", high_outlier_bounds)
    logging.info("Low outlier_bounds: %s", low_outlier_bounds)
    logging.info("total: %d", len(counting_ranges))

    ax = plt.gca()

    low_inlier = ~low_outlier
    high_inlier = ~high_inf & ~high_outlier

    xs = lows[low_inlier & high_inlier]
    ys = highs[low_inlier & high_inlier].astype(int)
    x_upper_line = np.max(xs) + 1
    y_upper_line = np.max(ys) + 1

    xbins = list(range(0, x_upper_line, binstep)) + [
        x_upper_line + e * binstep for e in [0, 2]
    ]
    ybins = list(range(0, y_upper_line, binstep)) + [
        y_upper_line + e * binstep for e in [0, 2, 3]
    ]
    # Main plot
    norm = mpl.colors.LogNorm()
    bins = (xbins, ybins)
    _, _, _, image = ax.hist2d(xs, ys, bins=bins, cmap="Blues", norm=norm)
    logging.info("low_inlier, high_inlier: %d", len(xs))

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

    ax.set_xticks([e for e in xbins[:-2] if e % tickstep == 0] + xbins[-2:-1])
    ax.set_xticks(xbins[:-2], minor=True)

    ax.set_yticks([e for e in ybins[:-3] if e % tickstep == 0] + ybins[-3:-1])
    ax.set_yticks(ybins[:-3], minor=True)
    yticklabels = ax.get_yticklabels()
    yticklabels[-1] = mpl.text.Text(text=r"$\infty$")
    ax.set_yticklabels(yticklabels)

    ax.set_xlabel("Low values")
    ax.set_ylabel("High values")
    ax.set_title("Ranges of counting operations")
    plt.savefig(output_dir / "hist2d.pdf")
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analysis pattern.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.output_dir)
