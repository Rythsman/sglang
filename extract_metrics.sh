#!/bin/bash
# Extract performance metrics from JSON file

if [ $# -lt 1 ]; then
    echo "Usage: $0 <json_file>"
    echo "Example: $0 input.json"
    exit 1
fi

json_file="$1"

if [ ! -f "$json_file" ]; then
    echo "Error: File '$json_file' not found"
    exit 1
fi

echo "Extracted Metrics:"
echo "--------------------------------------------------------------------------------"
echo -e "batch_size\tinput_len\toutput_len\tlatency\tttft\tinput_throughput\toutput_throughput"
echo "--------------------------------------------------------------------------------"

# Process each line of the JSON file
while IFS= read -r line; do
    # Skip empty lines
    if [ -z "$line" ]; then
        continue
    fi
    
    # Extract metrics using jq
    batch_size=$(echo "$line" | jq -r '.summary_info.batch_size')
    input_len=$(echo "$line" | jq -r '.summary_info.input_len')
    output_len=$(echo "$line" | jq -r '.summary_info.output_len')
    latency=$(echo "$line" | jq -r '.summary_info.latency')
    ttft=$(echo "$line" | jq -r '.summary_info.ttft')
    input_throughput=$(echo "$line" | jq -r '.summary_info.input_throughput')
    output_throughput=$(echo "$line" | jq -r '.summary_info.output_throughput')
    
    echo -e "${batch_size}\t${input_len}\t${output_len}\t${latency}\t${ttft}\t${input_throughput}\t${output_throughput}"
    
done < "$json_file"
