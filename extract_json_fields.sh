#!/bin/bash
# Extract specific fields from JSON lines file using bash/jq
# Usage: ./extract_json_fields.sh <input_file> [output_file]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_file> [output_file]"
    echo "If output_file is not specified, output will be printed to stdout."
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="${2:-}"

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed." >&2
    echo "Please install jq: sudo apt-get install jq (or equivalent)" >&2
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found." >&2
    exit 1
fi

# Function to extract fields from a JSON line
extract_line() {
    local line="$1"
    local batch_size=$(echo "$line" | jq -r '.summary_info.batch_size // "N/A"')
    local input_len=$(echo "$line" | jq -r '.summary_info.input_len // "N/A"')
    local output_len=$(echo "$line" | jq -r '.summary_info.output_len // "N/A"')
    local latency=$(echo "$line" | jq -r '.summary_info.latency // "N/A"')
    local ttft=$(echo "$line" | jq -r '.summary_info.ttft // "N/A"')
    local input_throughput=$(echo "$line" | jq -r '.summary_info.input_throughput // "N/A"')
    local output_throughput=$(echo "$line" | jq -r '.summary_info.output_throughput // "N/A"')
    
    echo "$batch_size,$input_len,$output_len,$latency,$ttft,$input_throughput,$output_throughput"
}

# Output to file or stdout
if [ -n "$OUTPUT_FILE" ]; then
    exec > "$OUTPUT_FILE"
fi

# Print CSV header
echo "batch_size,input_len,output_len,latency,ttft,input_throughput,output_throughput"

# Process each line
line_num=0
while IFS= read -r line || [ -n "$line" ]; do
    line_num=$((line_num + 1))
    # Skip empty lines
    if [ -z "$line" ]; then
        continue
    fi
    
    # Extract and output fields
    extract_line "$line" 2>/dev/null || echo "Error processing line $line_num" >&2
done < "$INPUT_FILE"

if [ -n "$OUTPUT_FILE" ]; then
    echo "Output written to $OUTPUT_FILE" >&2
fi
