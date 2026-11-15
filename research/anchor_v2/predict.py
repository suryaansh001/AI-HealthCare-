"""Command-line inference for a trusted ANCHOR v2 model artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from research.anchor_v2.inference import load_artifact, predict_one


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="UTF-8 JSON object with exactly the model's required feature keys",
    )
    arguments = parser.parse_args(argv)
    record = json.loads(arguments.input.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        parser.error("input JSON must be an object mapping feature names to values")
    artifact = load_artifact(arguments.model)
    print(json.dumps(predict_one(artifact, record), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
