"""Run checker"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import subprocess
import sys
from typing import Optional


def run_checker(pattern: str, timeout: int) -> Optional[str]:
    pattern = f"^(?:{pattern})$"
    try:
        cmd = [
            "java",
            "-jar",
            "third-party/checker.jar",
            "--cambiguity",
            "--witness",
            pattern,
        ]
        completed = subprocess.run(
            cmd, check=True, capture_output=True, timeout=timeout
        )
        return completed.stdout.decode(encoding="utf-8")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning(e)
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unambiguous", required=False, default="/dev/null")
    args = parser.parse_args()

    max_workers = 4
    timeout = 60

    def job(pattern: str) -> tuple[str, Optional[bool]]:
        output = run_checker(pattern, timeout)
        if output is None:
            return pattern, None

        is_ambiguous = output.splitlines(False)[-2].endswith("true")
        return pattern, is_ambiguous

    patterns = map(str.rstrip, sys.stdin)
    with open(args.unambiguous, "w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers) as executor:
            results = executor.map(job, patterns)
            for i, (pattern, is_ambiguous) in enumerate(results):
                if is_ambiguous is None:
                    logging.warning("Line %d: Checker fails to analyze", i)
                    continue
                if is_ambiguous:
                    print(pattern, flush=True)
                else:
                    print(pattern, flush=True, file=f)


if __name__ == "__main__":
    main()
