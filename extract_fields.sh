#!/bin/bash
# Extract specific fields from JSON file using jq (if available) or awk

if [ $# -lt 1 ]; then
    echo "Usage: $0 <json_file> [output_format]"
    echo "  output_format: csv (default) or table"
    exit 1
fi

JSON_FILE="$1"
OUTPUT_FORMAT="${2:-csv}"

if [ ! -f "$JSON_FILE" ]; then
    echo "Error: File '$JSON_FILE' not found." >&2
    exit 1
fi

# Check if jq is available
if command -v jq &> /dev/null; then
    # Use jq for JSON parsing
    if [ "$OUTPUT_FORMAT" = "csv" ]; then
        # Print CSV header
        echo "batch_size,input_len,output_len,latency,ttft,input_throughput,output_throughput"
        
        # Extract fields from each line
        while IFS= read -r line; do
            if [ -z "$line" ]; then
                continue
            fi
            
            batch_size=$(echo "$line" | jq -r '.summary_info.batch_size // ""')
            input_len=$(echo "$line" | jq -r '.summary_info.input_len // ""')
            output_len=$(echo "$line" | jq -r '.summary_info.output_len // ""')
            latency=$(echo "$line" | jq -r '.summary_info.latency // ""')
            ttft=$(echo "$line" | jq -r '.summary_info.ttft // ""')
            input_throughput=$(echo "$line" | jq -r '.summary_info.input_throughput // ""')
            output_throughput=$(echo "$line" | jq -r '.summary_info.output_throughput // ""')
            
            echo "$batch_size,$input_len,$output_len,$latency,$ttft,$input_throughput,$output_throughput"
        done < "$JSON_FILE"
    else
        # Print table header
        echo -e "batch_size\tinput_len\toutput_len\tlatency\tttft\tinput_throughput\toutput_throughput"
        
        # Extract fields from each line
        while IFS= read -r line; do
            if [ -z "$line" ]; then
                continue
            fi
            
            batch_size=$(echo "$line" | jq -r '.summary_info.batch_size // ""')
            input_len=$(echo "$line" | jq -r '.summary_info.input_len // ""')
            output_len=$(echo "$line" | jq -r '.summary_info.output_len // ""')
            latency=$(echo "$line" | jq -r '.summary_info.latency // ""')
            ttft=$(echo "$line" | jq -r '.summary_info.ttft // ""')
            input_throughput=$(echo "$line" | jq -r '.summary_info.input_throughput // ""')
            output_throughput=$(echo "$line" | jq -r '.summary_info.output_throughput // ""')
            
            echo -e "$batch_size\t$input_len\t$output_len\t$latency\t$ttft\t$input_throughput\t$output_throughput"
        done < "$JSON_FILE"
    fi
else
    echo "Error: jq is not installed. Please install jq or use the Python script instead." >&2
    echo "  Install: sudo apt-get install jq (Ubuntu/Debian)" >&2
    echo "  Or use: python extract_fields.py $JSON_FILE" >&2
    exit 1
fi
