#!/usr/bin/env python3
"""Extract specific fields from JSON file where each line is a JSON object."""

import json
import sys
import csv
from typing import List, Dict, Any


def extract_fields(json_file: str, output_format: str = "csv") -> None:
    """Extract specified fields from JSON file.

    Args:
        json_file: Path to input JSON file (one JSON object per line).
        output_format: Output format, either 'csv' or 'table'.
    """
    # Fields to extract from summary_info
    fields_to_extract = [
        "batch_size",
        "input_len",
        "output_len",
        "latency",
        "ttft",
        "input_throughput",
        "output_throughput",
    ]

    results = []

    # Read and parse each line
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    summary_info = data.get("summary_info", {})

                    # Extract required fields
                    row = {}
                    for field in fields_to_extract:
                        row[field] = summary_info.get(field, None)

                    results.append(row)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}", file=sys.stderr)
                    continue
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.", file=sys.stderr)
        sys.exit(1)

    # Output results
    if not results:
        print("No data extracted.", file=sys.stderr)
        return

    if output_format == "csv":
        # Output as CSV
        writer = csv.DictWriter(sys.stdout, fieldnames=fields_to_extract)
        writer.writeheader()
        writer.writerows(results)
    elif output_format == "table":
        # Output as formatted table
        print("\t".join(fields_to_extract))
        for row in results:
            print("\t".join(str(row.get(field, "")) for field in fields_to_extract))
    else:
        # Output as JSON lines
        for row in results:
            print(json.dumps(row))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_fields.py <json_file> [output_format]", file=sys.stderr)
        print("  output_format: csv (default), table, or json", file=sys.stderr)
        sys.exit(1)

    json_file = sys.argv[1]
    output_format = sys.argv[2] if len(sys.argv) > 2 else "csv"

    if output_format not in ["csv", "table", "json"]:
        print(f"Error: Invalid output format '{output_format}'. Use: csv, table, or json", file=sys.stderr)
        sys.exit(1)

    extract_fields(json_file, output_format)
