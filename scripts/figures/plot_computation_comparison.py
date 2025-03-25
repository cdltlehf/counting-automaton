"""Plot the range of counting operations."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, TextIO, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from parser_tools.constants import MAXREPEAT
from utils import escape
from utils import get_outlier_bounds
from utils.analysis import OutputDict

Range = tuple[int, Union[int, float]]


def operand_to_range(operand: Any) -> Range:
    low, high = operand
    return low, (high if high is not MAXREPEAT else float("inf"))


def main(
    input_file_x: TextIO,
    input_file_y: TextIO,
    output: Path,
    x_label: str,
    y_label: str,
) -> None:

    xs_list: list[float] = []
    ys_list: list[float] = []
    copied_list = []
    merged_list = []
    for json_object_x, json_object_y in zip(
        map(json.loads, input_file_x), map(json.loads, input_file_y)
    ):
        output_dict_x = OutputDict(**json_object_x)  # type: ignore
        output_dict_y = OutputDict(**json_object_y)  # type: ignore
        pattern = output_dict_x["pattern"]
        if pattern != output_dict_y["pattern"]:
            raise ValueError("Patterns do not match")
        results_x = output_dict_x["results"]
        results_y = output_dict_y["results"]
        if results_x is None or results_y is None:
            continue

        for test_case_result_x, test_case_result_y in zip(results_x, results_y):
            x = float("inf")
            x_merged = False
            x_copied = False

            y = float("inf")
            y_merged = False
            y_copied = False

            text = test_case_result_x["text"]
            if text != test_case_result_y["text"]:
                raise ValueError("Texts do not match")
            result_x = test_case_result_x["result"]
            result_y = test_case_result_y["result"]

            x_eval_symbol = None
            x_eval_predicate = None
            x_is_final = None

            y_eval_symbol = None
            y_eval_predicate = None
            y_is_final = None

            if result_x is not None:
                x_computation_info = result_x["computation_info"]
                x_eval_symbol = x_computation_info["EVAL_SYMBOL"]
                x_eval_predicate = x_computation_info["EVAL_PREDICATE"]
                x = sum(x_computation_info.values())
                x_copied = x_computation_info["ACCESS_NODE_CLONE"] > 0
                x_merged = x_computation_info["ACCESS_NODE_MERGE"] > 0
                x_is_final = result_x["is_final"]
            if result_y is not None:
                y_computation_info = result_y["computation_info"]
                y_eval_symbol = y_computation_info["EVAL_SYMBOL"]
                y_eval_predicate = y_computation_info["EVAL_PREDICATE"]
                y = sum(y_computation_info.values())
                y_copied = y_computation_info["ACCESS_NODE_CLONE"] > 0
                y_merged = y_computation_info["ACCESS_NODE_MERGE"] > 0
                y_is_final = result_y["is_final"]

            if result_x is not None and result_y is not None:
                assert x_eval_symbol is not None
                assert y_eval_symbol is not None
                assert x_eval_predicate is not None
                assert y_eval_predicate is not None
                assert x_is_final is not None
                assert y_is_final is not None

                if x_is_final != y_is_final:
                    # print(f"{escape(pattern)}\t{escape(text)}")
                    raise ValueError(
                        "is_final does not match. "
                        + f"{x_label}: {x_is_final}, {y_label}: {y_is_final}"
                    )

                if not (y_copied or y_merged or x_copied or x_merged):
                    if y_eval_symbol > x_eval_symbol:
                        # print(f"{escape(pattern)}\t{escape(text)}")
                        pass

            copied_list.append(x_copied or y_copied)
            merged_list.append(x_merged or y_merged)
            xs_list.append(x)
            ys_list.append(y)
    x_label = x_label.replace("_", " ")
    y_label = y_label.replace("_", " ")
    x_label = x_label[0].upper() + x_label[1:]
    y_label = y_label[0].upper() + y_label[1:]

    xs_total = np.array(xs_list)
    ys_total = np.array(ys_list)
    copied = np.array(copied_list)
    merged = np.array(merged_list)

    # scatter_lower_bound = np.percentile(np.concat([xs_total, ys_total]), 99)
    scatter_lower_bound = 500
    xs_over_scatter_lower_bound = xs_total >= scatter_lower_bound
    ys_over_scatter_lower_bound = ys_total >= scatter_lower_bound

    # xs = xs_total[xs_over_scatter_lower_bound | ys_over_scatter_lower_bound]
    # ys = ys_total[xs_over_scatter_lower_bound | ys_over_scatter_lower_bound]
    # lower_bound, upper_bound = get_outlier_bounds(np.concat([xs, ys]))
    lower_bound, upper_bound = 0, 10000

    xs_timeout = np.isinf(xs_total)
    ys_timeout = np.isinf(ys_total)
    xs_outlier = (
        (xs_total < lower_bound) | (xs_total >= upper_bound)
    ) & ~xs_timeout
    ys_outlier = (
        (ys_total < lower_bound) | (ys_total >= upper_bound)
    ) & ~ys_timeout

    ax = plt.gca()

    xs_inlier = ~xs_outlier & ~xs_timeout
    ys_inlier = ~ys_outlier & ~ys_timeout
    assert len(xs_total) == len(ys_total)
    cnt = 0
    logging.info("total: %d", len(xs_total))
    logging.info(f"{x_label} Inlier: %d", sum(xs_inlier))
    logging.info(f"{x_label} Outlier: %d", sum(xs_outlier))
    logging.info(f"{x_label} Timeout: %d", sum(xs_timeout))
    assert len(xs_total) == sum(xs_inlier) + sum(xs_outlier) + sum(xs_timeout)

    logging.info(f"{y_label} Inlier: %d", sum(ys_inlier))
    logging.info(f"{y_label} Outlier: %d", sum(ys_outlier))
    logging.info(f"{y_label} Timeout: %d", sum(ys_timeout))
    assert len(ys_total) == sum(ys_inlier) + sum(ys_outlier) + sum(ys_timeout)

    for x_type, x_indices in {
        f"{x_label} Inlier": xs_inlier,
        f"{x_label} Outlier": xs_outlier,
        f"{x_label} Timeout": xs_timeout,
    }.items():
        for y_type, y_indices in {
            f"{y_label} Inlier": ys_inlier,
            f"{y_label} Outlier": ys_outlier,
            f"{y_label} Timeout": ys_timeout,
        }.items():
            logging.info(f"{x_type}, {y_type}: %d", sum(x_indices & y_indices))
            cnt += sum(x_indices & y_indices)
    assert cnt == len(xs_total)

    # Main plot
    norm = mpl.colors.LogNorm()
    xbins = np.linspace(0, upper_bound, 21, dtype=int)
    steps = xbins[1] - xbins[0]
    x_max = xbins[-1]
    xbins = np.append(xbins, [x_max + steps, x_max + 2 * steps])
    ybins = xbins
    bins = (xbins, ybins)

    xs = xs_total[:]
    ys = ys_total[:]
    xs[xs_outlier] = xbins[-3]
    ys[ys_outlier] = ybins[-3]
    xs[xs_timeout] = xbins[-2]
    ys[ys_timeout] = ybins[-2]
    _, _, _, image = ax.hist2d(xs, ys, cmap="Blues", norm=norm, bins=bins)

    copied_or_merged = copied | merged
    over_scatter_lower_bound = (
        xs_over_scatter_lower_bound | ys_over_scatter_lower_bound
    )
    # The sparse counter config shows more dots than the bounded counter config
    # due to the timeouted results
    xs = xs_total[~copied_or_merged & over_scatter_lower_bound]
    ys = ys_total[~copied_or_merged & over_scatter_lower_bound]
    xs[xs > upper_bound] = upper_bound
    ys[ys > upper_bound] = upper_bound
    ax.scatter(xs, ys, color="red", marker=".", s=10, linewidth=1.0)

    ax.set_aspect("equal")
    plt.colorbar(image, ax=ax)

    linewidth = mpl.rcParams["xtick.major.width"]
    xmin = xbins[0]
    xmax = xbins[-1]
    for x in xbins[-3:-1]:
        ax.hlines(
            x, xmin, xmax, color="black", linestyles="--", linewidth=linewidth
        )
    ymin = ybins[0]
    ymax = ybins[-1]
    for y in ybins[-3:-1]:
        ax.vlines(
            y, ymin, ymax, color="black", linestyles="--", linewidth=linewidth
        )

    tickstep = 4
    ax.set_xticks([e for e in xbins[:-3:tickstep]] + list(xbins[-3:-1]))
    ax.set_xticks(xbins[:-3], minor=True)
    ax.set_yticks([e for e in ybins[:-3:tickstep]] + list(ybins[-3:-1]))
    ax.set_yticks(ybins[:-3], minor=True)
    xticklabels = [str(e) for e in xbins[:-3:tickstep]]
    xticklabels += [str(upper_bound), ""]
    yticklabels = [str(e) for e in ybins[:-3:tickstep]]
    yticklabels += [str(upper_bound), "Timeout"]
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Number of operations during computation")

    plt.savefig(output)
    plt.clf()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "computation_info_x",
        type=argparse.FileType("r"),
    )
    parser.add_argument(
        "computation_info_y",
        type=argparse.FileType("r"),
    )
    parser.add_argument("--x-label", type=str, required=True)
    parser.add_argument("--y-label", type=str, required=True)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(
        args.computation_info_x,
        args.computation_info_y,
        args.output,
        args.x_label,
        args.y_label,
    )
