"""Post processing"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
    result = entry["results"][0].get("result", {})
    if result is None:
        return dict(
            pattern=entry["pattern"],
            text=text,
            computation_steps=np.nan,
            matching_time=np.nan,
        )

    result = result.get("computation_info", {})
    return dict(
        pattern=entry["pattern"],
        text=text,
        computation_steps=get_computation_steps_from_results(result),
        matching_time=result["time_perf"],
    )


def load_dataframe(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        dataset = [preprocess(json.loads(e)) for e in f]
        dataset = list(filter(None, dataset))
        return pd.DataFrame(dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", type=Path)
    args = parser.parse_args()
    df = load_dataframe(args.analysis)
    print(df)


if __name__ == "__main__":
    main()
