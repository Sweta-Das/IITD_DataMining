#!/bin/bash

# A script to run Task 1, adhering strictly to the assignment's auto-grading requirements.
# This version uses explicit 'date' timing and strong process isolation for robustness.

# --- 1. Local Compilation (for convenience) ---
echo "INFO: Compiling local source code..."
# Ensure these paths are correct relative to where you run q1_1.sh (i.e., 'q1' folder)
(cd ./apriori/apriori/src/ && make >/dev/null 2>&1)
(cd ./fpgrowth/fpgrowth/src/ && make >/dev/null 2>&1)

if ! [[ -f "./apriori/apriori/src/apriori" ]]; then
    echo "WARNING: Local compilation of Apriori failed or executable not found locally. Proceeding with provided path..."
fi
if ! [[ -f "./fpgrowth/fpgrowth/src/fpgrowth" ]]; then
    echo "WARNING: Local compilation of FP-Growth failed or executable not found locally. Proceeding with provided path..."
fi
echo "INFO: Compilation step finished."


# --- 2. Argument Validation ---
if [ "$#" -ne 4 ]; then
    echo "ERROR: Invalid number of arguments provided."
    echo "Usage: $0 <path_apriori_executable> <path_fp_executable> <path_dataset> <path_out>"
    exit 1
fi

APRIORI_EXEC="$1"
FPGROWTH_EXEC="$2"
DATASET_PATH="$3"
OUTPUT_DIR="$4"

# --- 3. Verify Executables and Dataset Exist (CRITICAL) ---
if ! [[ -f "$APRIORI_EXEC" && -x "$APRIORI_EXEC" ]]; then
    echo "CRITICAL ERROR: Apriori executable not found or not executable at '$APRIORI_EXEC'."
    exit 1
fi
if ! [[ -f "$FPGROWTH_EXEC" && -x "$FPGROWTH_EXEC" ]]; then
    echo "CRITICAL ERROR: FP-Growth executable not found or not executable at '$FPGROWTH_EXEC'."
    exit 1
fi
if ! [[ -f "$DATASET_PATH" ]]; then
    echo "CRITICAL ERROR: Dataset file not found at '$DATASET_PATH'."
    exit 1
fi
echo "INFO: All provided executables and dataset found."


# --- 4. Setup Output Directory and CSV ---
mkdir -p "$OUTPUT_DIR"
SUPPORT_THRESHOLDS=(5 10 25 50 90)
RUNTIME_CSV="results.csv"

echo "SupportThreshold,Algorithm,RunTime(s),Status" > "$RUNTIME_CSV"

echo "INFO: Starting Task 1 experiments..."
echo "INFO: Output files will be saved in '$OUTPUT_DIR'"


# --- 5. Experiment Execution Loop (Timeout & Explicit Timing with Subshell Isolation) ---
for support in "${SUPPORT_THRESHOLDS[@]}"; do
    echo "INFO: Running with support threshold: $support%"

    # --- Apriori Run ---
    echo "  -> Apriori..."
    op_file_apriori="$OUTPUT_DIR/ap${support}"
    
    APRIORI_TIMEOUT_VAL_SEC="" # Store as number for bc, add 's' for timeout command
    if [[ "$support" -eq 5 ]]; then
        APRIORI_TIMEOUT_VAL_SEC="3600" # 1 hour
    elif [[ "$support" -eq 10 ]]; then
        APRIORI_TIMEOUT_VAL_SEC="3600" # 1 hour
    elif [[ "$support" -eq 25 ]]; then
        APRIORI_TIMEOUT_VAL_SEC="1800" # 30 minutes
    elif [[ "$support" -eq 50 ]]; then
        APRIORI_TIMEOUT_VAL_SEC="1800" # 30 minutes
    else # 90% support
        APRIORI_TIMEOUT_VAL_SEC="900"  # 15 minutes
    fi

    START_TIME=$(date +%s.%N)
    gtimeout "${APRIORI_TIMEOUT_VAL_SEC}s" \
        bash -c "\"$APRIORI_EXEC\" -s$support \"$DATASET_PATH\" \"$op_file_apriori\" 1>/dev/null 2>&1"
    APRIORI_EXIT_STATUS=$?
    END_TIME=$(date +%s.%N)

    RUNTIME=$(echo "$END_TIME - $START_TIME" | bc)
    STATUS="Completed"

    if [[ "$APRIORI_EXIT_STATUS" -eq 124 ]]; then
        RUNTIME="$APRIORI_TIMEOUT_VAL_SEC"
        STATUS="TimedOut"
        echo "WARNING: Apriori for ${support}% support timed out after ${RUNTIME}s."
    elif [[ "$APRIORI_EXIT_STATUS" -ne 0 ]]; then
        RUNTIME="0.000"
        STATUS="Error (Exit ${APRIORI_EXIT_STATUS})"
        echo "ERROR: Apriori for support $support% exited with status $APRIORI_EXIT_STATUS (not a timeout)."
    fi
    echo "$support,Apriori,$RUNTIME,$STATUS" >> "$RUNTIME_CSV"


    # --- FP-Growth Run ---
    echo "  -> FP-Growth..."
    op_file_fp="$OUTPUT_DIR/fp${support}"

    FPGROWTH_TIMEOUT_VAL_SEC="1800" # 30 minutes
    
    START_TIME=$(date +%s.%N)
    # FIX: Changed 'timeout' to 'gtimeout' for macOS compatibility
    gtimeout "${FPGROWTH_TIMEOUT_VAL_SEC}s" \
        bash -c "\"$FPGROWTH_EXEC\" -s $support \"$DATASET_PATH\" \"$op_file_fp\" 1>/dev/null 2>&1"
    FPGROWTH_EXIT_STATUS=$?
    END_TIME=$(date +%s.%N)

    RUNTIME=$(echo "$END_TIME - $START_TIME" | bc)
    STATUS="Completed"

    if [[ "$FPGROWTH_EXIT_STATUS" -eq 124 ]]; then
        RUNTIME="$FPGROWTH_TIMEOUT_VAL_SEC"
        STATUS="TimedOut"
        echo "WARNING: FP-Growth for ${support}% support timed out after ${RUNTIME}s."
    elif [[ "$FPGROWTH_EXIT_STATUS" -ne 0 ]]; then
        RUNTIME="0.000"
        STATUS="Error (Exit ${FPGROWTH_EXIT_STATUS})"
        echo "ERROR: FP-Growth for support $support% exited with status $FPGROWTH_EXIT_STATUS (not a timeout)."
    fi
    echo "$support,FP-Growth,$RUNTIME,$STATUS" >> "$RUNTIME_CSV"
done

echo "INFO: All experiments have been completed."
echo "INFO: Runtime data has been saved to '$RUNTIME_CSV'."


# --- 6. Plotting (Commented out) ---
# The final plot must be named 'plot.png' as per the assignment instructions.
# You will need 'generate_plot.py' in the same directory as this script.
# echo "INFO: Generating plot..."
# python generate_plot.py "$RUNTIME_CSV" "$OUTPUT_DIR/plot.png"

echo "INFO: Task 1 data generation completed successfully."