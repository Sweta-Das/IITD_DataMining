#!/bin/bash
if [ "$#" -ne 5 ]; then
    echo "Usage: bash q2.sh <path_gspan_executable> <path_fsg_executable> <path_gaston_executable> <path_dataset> <path_out>"
    exit 1
fi

GSPAN_EXE="$1"
FSG_EXE="$2"
GASTON_EXE="$3"
DATASET="$4"
OUTPUT_DIR="$5"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=========================================="
echo "Frequent Subgraph Mining Comparison"
echo "=========================================="
echo "gSpan executable: $GSPAN_EXE"
echo "FSG executable: $FSG_EXE"
echo "Gaston executable: $GASTON_EXE"
echo "Dataset: $DATASET"
echo "Output directory: $OUTPUT_DIR"
echo "Script directory: $SCRIPT_DIR"
echo "=========================================="
echo ""

if [ ! -f "$GSPAN_EXE" ]; then
    echo "Error: gSpan executable not found at $GSPAN_EXE"
    exit 1
fi

if [ ! -f "$FSG_EXE" ]; then
    echo "Error: FSG executable not found at $FSG_EXE"
    exit 1
fi

if [ ! -f "$GASTON_EXE" ]; then
    echo "Error: Gaston executable not found at $GASTON_EXE"
    exit 1
fi

if [ ! -f "$DATASET" ]; then
    echo "Error: Dataset not found at $DATASET"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

TEMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TEMP_DIR"
echo ""

echo "Step 1: Converting dataset formats..."
echo "--------------------------------------"
GSPAN_DATASET="$TEMP_DIR/dataset_gspan.txt"
FSG_DATASET="$TEMP_DIR/dataset_fsg.txt"

python3 "$SCRIPT_DIR/convert_dataset.py" "$DATASET" "$GSPAN_DATASET" "$FSG_DATASET"

if [ $? -ne 0 ]; then
    echo "Error: Dataset conversion failed"
    rm -rf "$TEMP_DIR"
    exit 1
fi
echo ""

echo "Step 2: Running algorithms at different support levels..."
echo "----------------------------------------------------------"
python3 "$SCRIPT_DIR/run_algorithms.py" "$GSPAN_EXE" "$FSG_EXE" "$GASTON_EXE" "$GSPAN_DATASET" "$FSG_DATASET" "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Algorithm execution failed"
    rm -rf "$TEMP_DIR"
    exit 1
fi
echo ""

echo "Step 3: Generating performance comparison plot..."
echo "--------------------------------------------------"
RESULTS_JSON="$OUTPUT_DIR/timing_results.json"
PLOT_FILE="$OUTPUT_DIR/plot.png"

python3 "$SCRIPT_DIR/plot_results.py" "$RESULTS_JSON" "$PLOT_FILE"

if [ $? -ne 0 ]; then
    echo "Error: Plot generation failed"
    rm -rf "$TEMP_DIR"
    exit 1
fi
echo ""

echo "Step 4: Cleaning up..."
echo "----------------------"
rm -rf "$TEMP_DIR"
echo "Removed temporary directory"
echo ""

echo "=========================================="
echo "All tasks completed successfully!"
echo "=========================================="
echo "Output files are located in: $OUTPUT_DIR"
echo "  - Algorithm outputs: gspan5, gspan10, ..., fsg5, fsg10, ..., gaston5, gaston10, ..."
echo "  - Timing results: timing_results.json"
echo "  - Performance plot: plot.png"
echo "=========================================="
