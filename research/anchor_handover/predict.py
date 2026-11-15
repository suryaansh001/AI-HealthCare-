"""CLI for strict ANCHOR backend-handover inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from research.anchor_handover.inference import DEFAULT_MODEL_PATH, load_model, predict_one


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON numeric constant is not allowed: {value}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="UTF-8 JSON object containing all 22 schema fields",
    )
    arguments = parser.parse_args(argv)
    record: Any = json.loads(
        arguments.input.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
    )
    if not isinstance(record, dict):
        parser.error("input JSON must be an object mapping feature names to values")
    artifact = load_model(arguments.model)
    result = predict_one(artifact, record)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
