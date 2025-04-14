"""scripts/preprocessing/preprocess_polyglot.py"""

import json
import os
from pathlib import Path

source_dir = Path("./raw-data/polyglot")
source_paths = [
    source_dir / "all_regexes.jsonl",
    # source_dir / "sl_regexes.jsonl",
]

target_dir = Path("./data/patterns")
target_paths = [
    target_dir / "all_regexes.txt",
    # target_dir / "sl_regexes.txt",
]

for source_path, target_path in zip(source_paths, target_paths):
    os.makedirs(target_dir.parent, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        for line in open(source_path, encoding="cp949"):
            try:
                obj = json.loads(line)
                pattern = obj["pattern"]
                f.write(json.dumps(pattern, indent=2) + "\n")
            except UnicodeEncodeError:
                pass
