#!/usr/bin/env python3
"""Send one or more images to the D-FINE LA HTTP API."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000/inference")
    parser.add_argument("--confidence", type=float)
    parser.add_argument(
        "--class-confidences",
        help=(
            "JSON object keyed by numeric class ID; for example "
            "'{\"4\":0.7,\"8\":0.3}' sets Page-footer and Table overrides. "
            "Unspecified classes inherit --confidence."
        ),
    )
    args = parser.parse_args()

    data = {}
    if args.confidence is not None:
        data["confidence"] = str(args.confidence)
    if args.class_confidences is not None:
        data["class_confidences"] = args.class_confidences

    with ExitStack() as stack:
        files = [
            (
                "images",
                (path.name, stack.enter_context(path.open("rb")), "application/octet-stream"),
            )
            for path in args.images
        ]
        response = requests.post(args.url, files=files, data=data, timeout=120)
    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False))


if __name__ == "__main__":
    main()
