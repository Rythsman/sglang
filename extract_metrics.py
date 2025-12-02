#!/usr/bin/env python3
"""Extract performance metrics from JSON file."""

import json
import sys
import csv


def extract_metrics(json_file, output_csv=None):
    """Extract specified metrics from JSON file.
    
    Args:
        json_file: Path to input JSON file (one JSON object per line)
        output_csv: Path to output CSV file (optional)
    """
    # Metrics to extract
    metrics = [
        "batch_size",
        "input_len",
        "output_len",
        "latency",
        "ttft",
        "input_throughput",
        "output_throughput"
    ]
    
    results = []
    
    # Read JSON file line by line
    with open(json_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                summary = data.get('summary_info', {})
                
                # Extract metrics
                row = {}
                for metric in metrics:
                    row[metric] = summary.get(metric, 'N/A')
                
                results.append(row)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}", file=sys.stderr)
                continue
    
    # Print results to console
    print("Extracted Metrics:")
    print("-" * 80)
    
    # Print header
    header = "\t".join(metrics)
    print(header)
    print("-" * 80)
    
    # Print data rows
    for row in results:
        values = [str(row[m]) for m in metrics]
        print("\t".join(values))
    
    # Save to CSV if output file is specified
    if output_csv:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=metrics)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✓ Results saved to: {output_csv}")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_metrics.py <json_file> [output_csv]")
        print("Example: python extract_metrics.py input.json output.csv")
        sys.exit(1)
    
    json_file = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    
    extract_metrics(json_file, output_csv)
