#!/bin/bash

# Local Compilation
echo "INFO: Compiling source code..."
(cd ./apriori/apriori/src/ && make >/dev/null 2>&1)
(cd ./fpgrowth/fpgrowth/src/ && make >/dev/null 2>&1)

# Check if local compilation was successful
if ! [[ -f "./apriori/apriori/src/apriori" ]]; then
    echo "WARNING: Local compilation of Apriori failed. Proceeding with provided path..."
fi
if ! [[ -f "./fpgrowth/fpgrowth/src/fpgrowth" ]]; then
    echo "WARNING: Local compilation of FP-Growth failed. Proceeding with provided path..."
fi
echo "INFO: Compilation completed."

# Arg Validation
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <path_apriori_executable> <path_fp_executable> <path_dataset> <path_out>"
    exit 1
fi

# Assign args to var names 
APRIORI_EXEC="$1"
FPGROWTH_EXEC="$2"
DATASET="$3"
OUTPUT_DIR="$4"

# Create output dir. '-p' flag prevents errors if it already exists
mkdir -p "$OUTPUT_DIR"

# Support Thresholds and Results File
RUNTIME_CSV="results.csv"
SUPPORT_THRESHOLDS=(5 10 25 50 90)

# Initialize CSV file with a header
echo "SupportThreshold,Algorithm,RunTime(s)" > "$RUNTIME_CSV"

# Compile the source code
echo "INFO: Starting Task-1.1..."
echo "INFO: Output files will be saved to '$OUTPUT_DIR'"


for support in "${SUPPORT_THRESHOLDS[@]}"; do
    echo "INFO: Running with support threshold: $support%"

    echo "  -> Apriori..."
    op_file_apriori="$OUTPUT_DIR/ap${support}"
    RUNTIME=$( { TIMEFORMAT='%R'; time $APRIORI_EXEC -s$support "$DATASET" "$op_file_apriori" 1>/dev/null 2>&1; } 2>&1 )
    echo "$support,Apriori,$RUNTIME" >> "$RUNTIME_CSV"

    echo "  -> FP-Growth..."
    op_file_fp="$OUTPUT_DIR/fp${support}"
    RUNTIME=$( { TIMEFORMAT='%R'; time $FPGROWTH_EXEC -s $support "$DATASET" "$op_file_fp" 1>/dev/null 2>&1; } 2>&1 )
    echo "$support,FP-Growth,$RUNTIME" >> "$RUNTIME_CSV"
done

echo "Runtime data saved to '$RUNTIME_CSV'"

# Plotting
echo "INFO: Generating plots..."
python generate_plot.py "$RUNTIME_CSV" "$OUTPUT_DIR/plot.png"

echo "INFO: Task-1.1 completed."
