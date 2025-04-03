"""Plot the range of counting operations."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable, TextIO, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from cai4py.counting_automaton.logging import ComputationStep
from cai4py.parser_tools.constants import MAXREPEAT
from cai4py.utils import escape
from cai4py.utils import get_outlier_bounds  # pylint: disable=unused-import
from cai4py.utils.analysis import OutputDict

Range = tuple[int, Union[int, float]]


def operand_to_range(operand: Any) -> Range:
    low, high = operand
    return low, (high if high is not MAXREPEAT else float("inf"))


def results_to_arrays(
    filtered_results: list[OutputDict],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool], npt.NDArray[np.bool]]:
    xs: list[float] = []
    copied_list: list[bool] = []
    merged_list: list[bool] = []
    for output_dict in filtered_results:
        results = output_dict["results"]
        if results is None:
            continue

        for test_case_result in results:
            x = float("inf")
            merged = False
            copied = False
            result = test_case_result["result"]
            if result is not None:
                computation_info = result["computation_info"]
                x = sum(
                    value
                    for key, value in computation_info.items()
                    if key in ComputationStep.__members__
                )
                copied = computation_info.get("ACCESS_NODE_CLONE", 0) > 0
                merged = computation_info.get("ACCESS_NODE_MERGE", 0) > 0

            copied_list.append(copied)
            merged_list.append(merged)
            xs.append(x)

    return np.array(xs), np.array(copied_list), np.array(merged_list)


def filter_analysis(
    x_objects: Iterable[object], y_objects: Iterable[object]
) -> tuple[list[OutputDict], list[OutputDict]]:
    x_filtered_results: list[OutputDict] = []
    y_filtered_results: list[OutputDict] = []
    for x_json_object, y_json_object in zip(x_objects, y_objects):
        x_output_dict = OutputDict(**x_json_object)  # type: ignore
        y_output_dict = OutputDict(**y_json_object)  # type: ignore
        pattern = x_output_dict["pattern"]
        if pattern != y_output_dict["pattern"]:
            raise ValueError("Patterns do not match")
        x_results = x_output_dict["results"]
        y_results = y_output_dict["results"]

        # If one of the results is None, the other should be None as well
        # if x_results is None or y_results is None:
        #     x_output_dict["results"] = None
        #     y_output_dict["results"] = None

        assert (x_results is None) == (y_results is None)

        if x_results is None or y_results is None:
            continue

        x_len = len(x_results)
        y_len = len(y_results)

        if x_len < y_len:
            x_results = x_results + [
                {"text": e["text"], "result": None} for e in y_results[x_len:]
            ]
        elif x_len > y_len:
            y_results = y_results + [
                {"text": e["text"], "result": None} for e in x_results[y_len:]
            ]

        x_output_dict["results"] = x_results
        y_output_dict["results"] = y_results

        # Assertion or debugging
        for x_test_case_result, y_test_case_result in zip(x_results, y_results):
            text = x_test_case_result["text"]
            if text != y_test_case_result["text"]:
                raise ValueError("Texts do not match")
            x_result = x_test_case_result["result"]
            y_result = y_test_case_result["result"]

            if x_result is not None and y_result is not None:
                x_is_final = x_result["is_final"]
                y_is_final = y_result["is_final"]

                if x_is_final != y_is_final:
                    print(f"{escape(pattern)}\t{escape(text)}")
                    raise ValueError("Final does not match")

        x_filtered_results.append(x_output_dict)
        y_filtered_results.append(y_output_dict)

    return x_filtered_results, y_filtered_results


def main(
    input_file_x: TextIO,
    input_file_y: TextIO,
    output: Path,
    x_label: str,
    y_label: str,
) -> None:

    x_label = x_label.replace("_", " ")
    y_label = y_label.replace("_", " ")
    x_label = x_label[0].upper() + x_label[1:]
    y_label = y_label[0].upper() + y_label[1:]

    filtered_results_x, filtered_results_y = filter_analysis(
        map(json.loads, input_file_x),
        map(json.loads, input_file_y),
    )

    xs_total, x_copied, x_merged = results_to_arrays(filtered_results_x)
    ys_total, y_copied, y_merged = results_to_arrays(filtered_results_y)

    copied = x_copied | y_copied
    merged = x_merged | y_merged
    copied_or_merged = copied | merged

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

    xs_inlier = ~xs_outlier & ~xs_timeout
    ys_inlier = ~ys_outlier & ~ys_timeout
    assert len(xs_total) == len(ys_total)

    logging.info("total: %d", len(xs_total))
    logging.info(f"{x_label} Inlier: %d", sum(xs_inlier))
    logging.info(f"{x_label} Outlier: %d", sum(xs_outlier))
    logging.info(f"{x_label} Timeout: %d", sum(xs_timeout))
    assert len(xs_total) == sum(xs_inlier) + sum(xs_outlier) + sum(xs_timeout)

    logging.info(f"{y_label} Inlier: %d", sum(ys_inlier))
    logging.info(f"{y_label} Outlier: %d", sum(ys_outlier))
    logging.info(f"{y_label} Timeout: %d", sum(ys_timeout))
    assert len(ys_total) == sum(ys_inlier) + sum(ys_outlier) + sum(ys_timeout)

    cnt = 0
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

            inlier = xs_inlier & ys_inlier
            if x_type == f"{x_label} Inlier" and y_type == f"{y_label} Inlier":
                logging.info(
                    "Inlier (xs > ys): %d",
                    sum(xs_total[inlier] > ys_total[inlier]),
                )

    assert cnt == len(xs_total)

    logging.info("Non-copying: %d", sum(~copied_or_merged))
    logging.info(
        f"{x_label} Inlier, Non-copying: %d", sum(xs_inlier & ~copied_or_merged)
    )
    logging.info(
        f"{x_label} Outlier, Non-copying: %d",
        sum(xs_outlier & ~copied_or_merged),
    )
    logging.info(
        f"{x_label} Timeout, Non-copying: %d",
        sum(xs_timeout & ~copied_or_merged),
    )

    logging.info(
        f"{y_label} Inlier, Non-copying: %d", sum(ys_inlier & ~copied_or_merged)
    )
    logging.info(
        f"{y_label} Outlier, Non-copying: %d",
        sum(ys_outlier & ~copied_or_merged),
    )
    logging.info(
        f"{y_label} Timeout, Non-copying: %d",
        sum(ys_timeout & ~copied_or_merged),
    )
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
            logging.info(
                f"{x_type}, {y_type}, Non-copying: %d",
                sum(x_indices & y_indices & ~copied_or_merged),
            )
            if x_type == f"{x_label} Inlier" and y_type == f"{y_label} Inlier":
                inlier = x_indices & y_indices & ~copied_or_merged
                logging.info(
                    "Inlier (xs > ys): %d",
                    sum(xs_total[inlier] > ys_total[inlier]),
                )

    # plt.figure(figsize=(6, 4))
    plt.figure(figsize=(4, 8 / 3))
    ax = plt.gca()

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

    ax.hlines(
        scatter_lower_bound,
        0,
        scatter_lower_bound,
        color="black",
        linestyles="--",
        linewidth=linewidth,
    )
    ax.vlines(
        scatter_lower_bound,
        0,
        scatter_lower_bound,
        color="black",
        linestyles="--",
        linewidth=linewidth,
    )

    over_scatter_lower_bound = (
        xs_over_scatter_lower_bound | ys_over_scatter_lower_bound
    )
    # The sparse counter config shows more dots than the bounded counter config
    # due to the timeouted results
    scatter_xs = xs[~copied_or_merged & over_scatter_lower_bound]
    scatter_ys = ys[~copied_or_merged & over_scatter_lower_bound]
    ax.scatter(
        scatter_xs,
        scatter_ys,
        color="red",
        marker="o",
        s=5,
        linewidth=1.0,
        alpha=0.5,
    )

    tickstep = 4
    ax.set_xticks([e for e in xbins[:-3:tickstep]] + list(xbins[-3:-1]))
    ax.set_xticks(xbins[:-3], minor=True)
    ax.set_yticks([e for e in ybins[:-3:tickstep]] + list(ybins[-3:-1]))
    ax.set_yticks(ybins[:-3], minor=True)
    xticklabels = [str(e) for e in xbins[:-3:tickstep]]
    xticklabels += [str(upper_bound), ""]
    yticklabels = [str(e) for e in ybins[:-3:tickstep]]
    yticklabels += ["", "Timeout"]
    ax.set_xticklabels(xticklabels, rotation=30, ha="right")
    ax.set_yticklabels(yticklabels)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.set_title("Number of operations")
    # plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
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
