#!/usr/bin/env python3
"""Merge multiple Chrome trace event files (JSON or JSON.GZ) into one.

This utility is intended for cases where different subsystems dump trace events
into separate files (e.g. main trace and NVML sampler trace), and users want to
combine them for viewing in Chrome trace viewer / Perfetto.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from typing import Any, List


def _load_events(path: str) -> List[dict]:
    """Load a list of Chrome trace events from a JSON(.gz) file."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            data = json.load(f)
    else:
        with open(path, "r") as f:
            data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Trace file is not a JSON list: {path}")
    # Filter out unexpected items defensively.
    return [e for e in data if isinstance(e, dict)]


def _dump_events(path: str, events: List[dict]) -> None:
    """Write events to a JSON(.gz) file."""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    payload = json.dumps(events, indent=4, separators=(",", ":"))
    if path.endswith(".gz"):
        with gzip.open(path, "wt") as f:
            f.write(payload)
    else:
        with open(path, "w") as f:
            f.write(payload)


def _event_sort_key(e: dict) -> Any:
    # Primary: timestamp (microseconds). Secondary: phase/name for stability.
    return (e.get("ts", 0), e.get("ph", ""), e.get("name", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Chrome trace JSON files.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output path (.json or .json.gz).",
    )
    parser.add_argument(
        "--sort-by-ts",
        action="store_true",
        help="Sort merged events by 'ts' (timestamp).",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input trace files (.json or .json.gz).",
    )
    args = parser.parse_args()

    merged: List[dict] = []
    for p in args.inputs:
        merged.extend(_load_events(p))

    if args.sort_by_ts:
        merged.sort(key=_event_sort_key)

    _dump_events(args.output, merged)


if __name__ == "__main__":
    main()

