#!/usr/bin/env python3
"""Extract specific fields from JSON lines file.

This script reads a JSON lines file where each line is a JSON object,
and extracts the following fields:
- batch_size
- input_len
- output_len
- latency
- ttft
- input_throughput
- output_throughput
"""

import json
import sys
from typing import Dict, Any, Optional


def extract_fields(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract required fields from JSON data.

    Args:
        data: JSON object containing the data.

    Returns:
        Dictionary with extracted fields, or None if extraction fails.
    """
    try:
        summary_info = data.get("summary_info", {})
        result = {
            "batch_size": summary_info.get("batch_size"),
            "input_len": summary_info.get("input_len"),
            "output_len": summary_info.get("output_len"),
            "latency": summary_info.get("latency"),
            "ttft": summary_info.get("ttft"),
            "input_throughput": summary_info.get("input_throughput"),
            "output_throughput": summary_info.get("output_throughput"),
        }
        return result
    except (KeyError, AttributeError) as e:
        print(f"Error extracting fields: {e}", file=sys.stderr)
        return None


def main():
    """Main function to process JSON lines file."""
    if len(sys.argv) < 2:
        print("Usage: python extract_json_fields.py <input_file> [output_file]")
        print("If output_file is not specified, output will be printed to stdout.")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            results = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    extracted = extract_fields(data)
                    if extracted:
                        results.append(extracted)
                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Failed to parse JSON on line {line_num}: {e}",
                        file=sys.stderr,
                    )
                    continue

        # Output results
        output = sys.stdout
        if output_file:
            output = open(output_file, "w", encoding="utf-8")

        try:
            # Output as CSV header
            header = "batch_size,input_len,output_len,latency,ttft,input_throughput,output_throughput"
            print(header, file=output)

            # Output data rows
            for result in results:
                row = (
                    f"{result['batch_size']},{result['input_len']},"
                    f"{result['output_len']},{result['latency']},"
                    f"{result['ttft']},{result['input_throughput']},"
                    f"{result['output_throughput']}"
                )
                print(row, file=output)

            # Also output as JSON if user wants
            if output_file:
                json_output_file = output_file.replace(".csv", ".json")
                if json_output_file == output_file:
                    json_output_file = output_file + ".json"
                with open(json_output_file, "w", encoding="utf-8") as json_f:
                    json.dump(results, json_f, indent=2)

        finally:
            if output_file and output != sys.stdout:
                output.close()

        print(f"Processed {len(results)} lines successfully.", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
