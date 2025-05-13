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
from cai4py.utils import get_outlier_bounds
from cai4py.utils.analysis import OutputDict

Range = tuple[int, Union[int, float]]


def operand_to_range(operand: Any) -> Range:
    low, high = operand
    return low, (high if high is not MAXREPEAT else float("inf"))


def results_to_arrays(
    filtered_results: list[OutputDict],
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.bool],
    npt.NDArray[np.bool],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    xs: list[float] = []
    copied_list: list[bool] = []
    merged_list: list[bool] = []
    max_num_keys_list: list[int] = []
    max_synchronization_degree_list: list[int] = []
    for output_dict in filtered_results:
        results = output_dict["results"]
        assert results is not None

        for test_case_result in results[:1]:
            x = float("inf")
            max_num_keys = 0
            max_synchronization_degree = 0
            marked_x = 0  # pylint: disable=unused-variable
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
                marked_x = sum(
                    value
                    for key, value in computation_info.items()
                    if key not in ComputationStep.__members__
                )
                max_num_keys = computation_info.get("max_num_keys", 0)
                max_synchronization_degree = computation_info.get(
                    "max_synchronization_degree", 0
                )
                copied = computation_info.get("ACCESS_NODE_CLONE", 0) > 0
                merged = computation_info.get("ACCESS_NODE_MERGE", 0) > 0

            max_num_keys_list.append(max_num_keys)
            max_synchronization_degree_list.append(max_synchronization_degree)
            copied_list.append(copied)
            merged_list.append(merged)
            # xs.append(x - marked_x)
            xs.append(x)

    return (
        np.array(xs),
        np.array(copied_list),
        np.array(merged_list),
        np.array(max_num_keys_list),
        np.array(max_synchronization_degree_list),
    )


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
    plt.rcParams.update({"text.usetex": True})

    x_label = x_label.replace("_", " ")
    y_label = y_label.replace("_", " ")
    x_label = x_label[0].upper() + x_label[1:]
    y_label = y_label[0].upper() + y_label[1:]

    filtered_results_x, filtered_results_y = filter_analysis(
        map(json.loads, input_file_x),
        map(json.loads, input_file_y),
    )

    # pylint: disable-next=unused-variable
    (xs_total, x_copied, x_merged, xs_num_key, xs_degree) = results_to_arrays(
        filtered_results_x
    )
    # pylint: disable-next=unused-variable
    (ys_total, y_copied, y_merged, ys_num_key, ys_degree) = results_to_arrays(
        filtered_results_y
    )

    xs_timeout = np.isinf(xs_total)
    ys_timeout = np.isinf(ys_total)
    lower_bound, upper_bound = get_outlier_bounds(
        np.concat([xs_total, ys_total])
    )

    xs_outlier = (
        (xs_total < lower_bound) | (xs_total >= upper_bound)
    ) & ~xs_timeout
    ys_outlier = (
        (ys_total < lower_bound) | (ys_total >= upper_bound)
    ) & ~ys_timeout

    xs_inlier = ~xs_outlier & ~xs_timeout
    ys_inlier = ~ys_outlier & ~ys_timeout
    assert len(xs_total) == len(ys_total)

    ############################################################################
    # Print the simple comparisons
    ############################################################################
    logging.info("total: %d", len(xs_total))
    ############################################################################
    indices = xs_total > ys_total
    logging.info(
        "x > y: %d (%0.2f)",
        sum(indices),
        sum(indices) / len(xs_total) * 100,
    )
    indices = xs_total < ys_total
    logging.info(
        "x < y: %d (%0.2f)",
        sum(indices),
        sum(indices) / len(xs_total) * 100,
    )
    ############################################################################
    indices = xs_total > 1.1 * ys_total
    logging.info(
        "x > 1.1 y: %d (%0.2f)",
        sum(indices),
        sum(indices) / len(xs_total) * 100,
    )
    indices = 1.1 * xs_total < ys_total
    logging.info(
        "1.1 x < y: %d (%0.2f)",
        sum(indices),
        sum(indices) / len(xs_total) * 100,
    )
    ############################################################################
    indices = xs_total > 1.5 * ys_total
    logging.info(
        "x > 1.5 y: %d (%0.2f)",
        sum(indices),
        sum(indices) / len(xs_total) * 100,
    )
    indices = 1.5 * xs_total < ys_total
    logging.info(
        "1.5 x < y: %d (%0.2f)",
        sum(indices),
        sum(indices) / len(xs_total) * 100,
    )
    ############################################################################
    indices = xs_total > 2.0 * ys_total
    logging.info(
        "x > 2.0 y: %d (%0.2f)",
        sum(indices),
        sum(indices) / len(xs_total) * 100,
    )
    indices = 1.0 * xs_total < ys_total
    logging.info(
        "2.0 x < y: %d (%0.2f)",
        sum(indices),
        sum(indices) / len(xs_total) * 100,
    )
    ############################################################################
    # not x_copied
    # indices = ~x_copied & ~y_copied
    # logging.info("non-copied: %d", sum(indices))
    # logging.info("x > y: %d", sum((xs_total > ys_total)[indices]))
    # logging.info("x < y: %d", sum((xs_total < ys_total)[indices]))

    # y degree <= 1
    # indices = ys_degree <= 1
    # logging.info("y degree <= 1: %d", sum(indices))
    # logging.info("x > y: %d", sum((xs_total > ys_total)[indices]))
    # logging.info("x < y: %d", sum((xs_total < ys_total)[indices]))
    ############################################################################

    ############################################################################
    # Print the pattern with the largest difference among inliers
    ############################################################################
    diff = np.where(xs_timeout, 0, xs_total) - np.where(ys_timeout, 0, ys_total)
    indices = xs_inlier & ys_inlier

    x_diff_max = np.argmax(np.where(~indices, 0, diff))

    logging.info("x >> y: %d", diff[x_diff_max])
    logging.info("degree: %d", ys_degree[x_diff_max])
    logging.info("num_key: %d", ys_num_key[x_diff_max])
    results_x = filtered_results_x[x_diff_max]
    logging.info(filtered_results_x[x_diff_max])
    logging.info(filtered_results_y[x_diff_max])

    pattern = results_x["pattern"]
    if results_x["results"] is not None:
        for results in results_x["results"]:
            text = results["text"]
            print(escape(pattern), escape(text), sep="\t")

    y_diff_max = np.argmax(np.where(~indices, 0, -diff))

    logging.info("x << y: %d", -diff[y_diff_max])
    logging.info("degree: %d", ys_degree[y_diff_max])
    logging.info("num_key: %d", ys_num_key[y_diff_max])
    results_y = filtered_results_y[y_diff_max]
    logging.info(filtered_results_x[y_diff_max])
    logging.info(filtered_results_y[y_diff_max])
    ############################################################################

    pattern = results_y["pattern"]
    if results_y["results"] is not None:
        for results in results_y["results"]:
            text = results["text"]
            print(escape(pattern), escape(text), sep="\t")

    ############################################################################
    # Print histogram information
    ############################################################################
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
    ############################################################################

    assert cnt == len(xs_total)

    plt.figure(figsize=(2, 2))
    ax = plt.gca()

    # Main plot
    norm = mpl.colors.LogNorm(vmin=0.1)
    xbins = np.linspace(0, upper_bound, 11, dtype=int)
    steps = xbins[1] - xbins[0]
    x_max = xbins[-1]
    xbins = np.append(xbins, [x_max + steps, x_max + 2 * steps])
    ybins = xbins
    bins = (xbins, ybins)

    xs = xs_total.copy()
    ys = ys_total.copy()
    xs[xs_outlier] = xbins[-3]
    ys[ys_outlier] = ybins[-3]
    xs[xs_timeout] = xbins[-2]
    ys[ys_timeout] = ybins[-2]

    ############################################################################
    indices = np.full_like(xs, True, dtype=bool)
    ############################################################################
    # indices = ys_degree <= 2
    # logging.info("2-sync: %d", sum(indices))
    ############################################################################
    # indices = ~x_copied & ~y_copied
    # logging.info("non-replicating: %d", sum(indices))
    ############################################################################

    _, _, _, image = ax.hist2d(
        xs[indices], ys[indices], cmap="Blues", norm=norm, bins=bins
    )
    ax.set_aspect("equal")
    cbar = plt.colorbar(image, ax=ax, shrink=0.8)
    vmax = cbar.mappable.norm.vmax
    cbar.ax.set_ylim(1, vmax)

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

    xticks = [xtick for xtick in ax.get_xticks() if xtick < upper_bound]
    xticklabels = ax.get_xticklabels()[: len(xticks)]
    xticks += [xbins[-3], xbins[-2]]
    xticklabels += [xbins[-3], ""]  # type: ignore
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    yticks = [ytick for ytick in ax.get_yticks() if ytick < upper_bound]
    yticklabels = ax.get_yticklabels()[: len(yticks)]
    yticks += [xbins[-3], xbins[-2]]
    yticklabels += ["", "T/O"]  # type: ignore
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    # ax.set_xlabel(x_label)
    # ax.set_ylabel(y_label)
    # ax.set_title("Number of operations")
    # plt.tight_layout()
    plt.savefig(output, bbox_inches="tight", dpi=300)
    plt.savefig(output.with_suffix(".pgf"), bbox_inches="tight", dpi=300)
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
