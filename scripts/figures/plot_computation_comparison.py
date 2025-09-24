"""Plot the computation comparison of configuration representation."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd


def get_outlier_bounds(data: npt.NDArray[np.float32]) -> tuple[float, float]:
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound


def get_computation_steps_from_results(result: dict[str, Any]) -> int | None:
    keys = [
        "EVAL_SYMBOL",
        "EVAL_PREDICATE",
        "APPLY_OPERATION",
        "ACCESS_NODE_MERGE",
        "ACCESS_NODE_CLONE",
    ]
    return int(sum(result.get(k, 0) for k in keys))


def preprocess(entry: dict[str, Any]) -> dict[str, Any] | None:
    assert entry["results"] is not None
    if not entry["results"]:
        return None

    text = entry["results"][0]["text"]
    result = entry["results"][0].get("result", None)
    if result is None:
        return dict(
            pattern=entry["pattern"],
            text=text,
            computation_steps=np.nan,
            matching_time=np.nan,
            result=None,
        )

    computation_info = result.get("computation_info", {})
    is_matched = result["is_final"]
    return dict(
        pattern=entry["pattern"],
        text=text,
        computation_steps=get_computation_steps_from_results(computation_info),
        matching_time=computation_info["time_perf"],
        is_matched=is_matched,
    )


def load_dataset(path: Path) -> list[dict[str, Any] | None]:
    with open(path, encoding="utf-8") as f:
        dataset = [preprocess(json.loads(e)) for e in f]
        return dataset


def hist2d(ax, xs: npt.NDArray, ys: npt.NDArray, **kwargs):  # type: ignore
    xs = xs.copy()
    ys = ys.copy()

    ax.set_aspect("equal")

    xs_timeout = np.isnan(xs)
    ys_timeout = np.isnan(ys)

    indices = ~np.isnan(xs) & ~np.isnan(ys)
    xlim = np.quantile(xs[indices], 0.99)
    ylim = np.quantile(ys[indices], 0.99)
    threshold = max(xlim, ylim)

    xs_outlier = (xs >= threshold) & ~xs_timeout
    ys_outlier = (ys >= threshold) & ~ys_timeout

    xbins = np.linspace(0, threshold, 11)
    step = xbins[-1] - xbins[-2]
    xbins = xbins = np.append(xbins, [threshold + step, threshold + 2 * step])

    outlier_placeholder = xbins[-2] - step / 2
    timeout_placeholder = xbins[-1] - step / 2
    ybins = xbins
    bins = (xbins, ybins)

    xs[xs_outlier] = outlier_placeholder
    ys[ys_outlier] = outlier_placeholder
    xs[xs_timeout] = timeout_placeholder
    ys[ys_timeout] = timeout_placeholder

    _, _, _, image = ax.hist2d(
        xs, ys, norm=mpl.colors.LogNorm(), bins=bins, **kwargs
    )
    ax.set_aspect("equal")
    plt.colorbar(image, ax=ax, shrink=0.8)
    # vmax = cbar.mappable.norm.vmax
    # cbar.ax.set_ylim(-1, vmax)

    linewidth = mpl.rcParams["xtick.major.width"]
    for x in xbins[-3:-1]:
        ax.hlines(
            x, 0, xbins[-1], color="black", linestyles="--", linewidth=linewidth
        )
    for y in ybins[-3:-1]:
        ax.vlines(
            y, 0, ybins[-1], color="black", linestyles="--", linewidth=linewidth
        )

    # xticks = [xtick for xtick in ax.get_xticks() if xtick < threshold]
    # xticklabels = ax.get_xticklabels()[: len(xticks)]
    xticks = [xbins[-3], xbins[-2]]
    xticklabels = [f"{float(xbins[-3]):.2f}", ""]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    yticks = [ytick for ytick in ax.get_yticks() if ytick < threshold]
    yticklabels = ax.get_yticklabels()[: len(yticks)]
    yticks += [xbins[-3], xbins[-2]]
    yticklabels += ["", "T/O"]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "property", type=str, choices=["computation-steps", "matching-time"]
    )
    parser.add_argument("computation_info_x", type=Path)
    parser.add_argument("computation_info_y", type=Path)
    parser.add_argument("--x-label", type=str, required=True)
    parser.add_argument("--y-label", type=str, required=True)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    # plt.rcParams.update({"text.usetex": True})

    x_label = args.x_label.replace("_", " ")
    y_label = args.y_label.replace("_", " ")
    x_label = x_label[0].upper() + x_label[1:]
    y_label = y_label[0].upper() + y_label[1:]

    x_dataset = load_dataset(args.computation_info_x)
    y_dataset = load_dataset(args.computation_info_y)
    assert len(x_dataset) == len(y_dataset)

    error = [x is None or y is None for x, y in zip(x_dataset, y_dataset)]

    x_dataset = [x for x, e in zip(x_dataset, error) if not e]
    y_dataset = [y for y, e in zip(y_dataset, error) if not e]
    print(len(x_dataset))
    print(len(y_dataset))

    x_df = pd.DataFrame(x_dataset)
    y_df = pd.DataFrame(y_dataset)

    if args.property == "computation-steps":
        xs = np.array(x_df.computation_steps)
        ys = np.array(y_df.computation_steps)
    elif args.property == "matching-time":
        xs = np.array(x_df.matching_time, dtype=float) / (10**9)
        ys = np.array(y_df.matching_time, dtype=float) / (10**9)
    else:
        exit(-1)

    ###########################################################################
    # Unexpected counter expansion vs
    ###########################################################################
    for entry_x, entry_y in zip(x_dataset, y_dataset):
        if entry_x["computation_steps"] > 3 * entry_y["computation_steps"]:
            print(json.dumps(entry_x))
            print(json.dumps(entry_y))
            print(
                json.dumps(
                    {
                        "pattern": entry_x["pattern"],
                        "texts": [entry_x["text"]],
                    }
                )
            )
    ###########################################################################

    plt.figure(figsize=(2, 2))
    ax = plt.gca()

    # plt.tight_layout()
    original_cmap = plt.get_cmap("Blues")
    start = 0.2
    new_colors = original_cmap(np.linspace(start, 1, int(256 * (1 - start))))
    cmap = ListedColormap(new_colors)
    hist2d(ax, xs, ys, cmap=cmap)
    # ax.set_xlabel(x_label)
    # ax.set_ylabel(y_label)
    # ax.set_title("Number of operations")

    plt.savefig(args.output, bbox_inches="tight", dpi=300)
    # plt.savefig(output.with_suffix(".pgf"), bbox_inches="tight", dpi=300)
    plt.clf()


if __name__ == "__main__":
    main()
