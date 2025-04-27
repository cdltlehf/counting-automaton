"""Case study for comparing the performance of different configs."""

import argparse
from collections import defaultdict as dd
import io
import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from matplotlib.lines import Line2D as L
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import timeout_decorator  # type: ignore

from cai4py.counting_automaton.logging import ComputationStep
from cai4py.counting_automaton.logging import ComputationStepMark
from cai4py.counting_automaton.logging import VERBOSE
import cai4py.counting_automaton.position_counting_automaton as pca
import cai4py.counting_automaton.super_config as sc

logger = logging.getLogger(__name__)


class VerboseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == VERBOSE


def collect_computation_info(
    automaton: pca.PositionCountingAutomaton,
    w: str,
    get_computation: Callable[
        [pca.PositionCountingAutomaton, str], Iterable[sc.SuperConfigBase]
    ],
) -> tuple[dict[str, int], bool]:
    logger_dict = logging.Logger.manager.loggerDict
    counting_automaton_loggers = {
        name: counting_automaton_logger
        for name, counting_automaton_logger in logger_dict.items()
        if name.startswith("cai4py")
        and isinstance(counting_automaton_logger, logging.Logger)
    }
    counting_automaton_logger_configs = {
        name: (logger.level, logger.handlers.copy())
        for name, logger in counting_automaton_loggers.items()
    }
    stream = io.StringIO()
    handler: logging.Handler = logging.StreamHandler(stream)
    logger_filter = VerboseFilter()
    handler.addFilter(logger_filter)

    for counting_automaton_logger in counting_automaton_loggers.values():
        counting_automaton_logger.setLevel(VERBOSE)
        counting_automaton_logger.propagate = False
        counting_automaton_logger.handlers.clear()
        counting_automaton_logger.addHandler(handler)

    computation_info: dd[str, int] = dd(int)
    last_super_config: Optional[sc.SuperConfigBase] = None
    try:
        mark_flags: dd[str, bool] = dd(bool)
        for i, super_config in enumerate(get_computation(automaton, w)):
            pos = stream.tell()
            value = stream.getvalue()[:pos]
            for computation_step in value.splitlines():
                if computation_step in ComputationStep.__members__:
                    computation_info[computation_step] += 1

                    for mark, flag in mark_flags.items():
                        if not flag:
                            continue
                        marked_computation_step = f"{mark}_{computation_step}"
                        computation_info[marked_computation_step] += 1

                elif computation_step in ComputationStepMark.__members__:
                    if computation_step.startswith("START_"):
                        computation_step = computation_step[6:]
                        if mark_flags[computation_step]:
                            raise ValueError(
                                f"Duplicate start mark: {computation_step}"
                            )
                        mark_flags[computation_step] = True
                    elif computation_step.startswith("END_"):
                        computation_step = computation_step[4:]
                        if not mark_flags[computation_step]:
                            raise ValueError(
                                f"Duplicate end mark: {computation_step}"
                            )
                        mark_flags[computation_step] = False
                else:
                    print(list(ComputationStepMark.__members__))
                    raise ValueError(
                        f"Unknown computation step: {computation_step}"
                    )
            stream.seek(0)
            stream.truncate(pos)
            last_super_config = super_config
            logger.debug("Computation info %d: %s", i, dict(computation_info))
        assert last_super_config is not None
    finally:
        handler.close()
        for counting_automaton_logger, (level, handlers) in zip(
            counting_automaton_loggers.values(),
            counting_automaton_logger_configs.values(),
        ):
            counting_automaton_logger.setLevel(level)
            counting_automaton_logger.handlers.clear()
            for handler in handlers:
                counting_automaton_logger.addHandler(handler)
    return dict(computation_info), last_super_config.is_final()


def collect_optional_computation_info(
    automaton: pca.PositionCountingAutomaton,
    w: str,
    get_computation: Callable[
        [pca.PositionCountingAutomaton, str], Iterable[sc.SuperConfigBase]
    ],
    timeout: int,
) -> Optional[dict[str, Any]]:
    try:
        computation_info, is_final = timeout_decorator.timeout(timeout)(
            collect_computation_info
        )(automaton, w, get_computation)
        return {"computation_info": computation_info, "is_final": is_final}
    except timeout_decorator.TimeoutError:
        logger.warning("Computation timeout when processing text %s", w[:100])
        return None


def main(output_dir: Path) -> None:
    timeout = 60
    k = 100

    get_computations = {
        "Super config": sc.BoundedSuperConfig.get_computation,
        "C-config": sc.BoundedCounterConfig.get_computation,
        "Sparse c-config": sc.SparseCounterConfig.get_computation,
        "Dc-config": sc.DeterminizedBoundedCounterConfig.get_computation,
    }
    plt.rcParams.update({"text.usetex": True})

    patterns = [
        (f"a*a{{{k}}}", f"$a^*a^{k}$"),
        (f"(aa*){{0,{k}}}", f"$(a^*a)^{{0,{k}}}$"),
        (f"a*(a|a){{{k}}}", f"$a^*(a|a)^{k}$"),
    ]
    ylim = (-400, 15000)
    for i, (pattern, _) in enumerate(patterns):
        logger.info("Pattern: %s", pattern)
        # output = output_dir / f"case_study_{i}.pdf"
        output = output_dir / f"case_study_{i}.pgf"
        plt.figure(figsize=(2.5, 2))
        ax = plt.gca()
        automaton = pca.PositionCountingAutomaton.create(pattern)
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        markers = ["o", "s", "D", "^", "v"]
        for j, (method, get_computation) in enumerate(get_computations.items()):
            logger.info("Method: %s", method)
            xs = []
            for n in range(1, 100):
                text = "a" * n
                result = collect_optional_computation_info(
                    automaton, text, get_computation, timeout
                )
                if result is None:
                    continue

                computation_info = result["computation_info"]
                x = sum(
                    value
                    for key, value in computation_info.items()
                    if key in ComputationStep.__members__
                )
                xs.append(x)
                if x > ylim[1]:
                    break
            markevery = range(j * 8, len(xs), 32)
            color = colors[j]
            marker = markers[j]
            ax.plot(xs, label=method, color=color)
            ax.plot(
                xs,
                label="_nolegend_",
                marker=marker,
                markevery=markevery,
                linestyle="",
                zorder=10,
                color=color,
            )

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax.yaxis.set_major_formatter(formatter)
        ax.grid(axis="y", alpha=0.5)
        ax.grid(axis="y", which="minor", alpha=0.3, linestyle="dashed")
        # ax.set_title(f"Pattern: {title}")
        # ax.set_xlabel("Length of text $a^k$")
        # ax.set_ylabel("Computation steps")
        if i == 0:
            legend_handles = [
                L([], [], color=color, marker=marker)
                for color, marker in zip(colors, markers)
            ]
            ax.legend(
                handles=legend_handles,
                labels=get_computations.keys(),
                loc="upper left",
            )
        ax.set_ylim(ylim)
        ax.minorticks_on()
        plt.savefig(output, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    main(args.output_dir)
