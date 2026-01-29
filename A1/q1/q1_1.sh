#!/bin/bash

# Configuration 
APRIORI_EXEC="./apriori/apriori/src/apriori"
FPGROWTH_EXEC="./fpgrowth/fpgrowth/src/fpgrowth"
DATASET="./dataset/webdocs.dat"
RESULTS_FILE="results.csv"
SUPPORT_THRESHOLDS=(5 10 25 50 90)

# Compile the source code
echo "INFO: Compiling source code..."
(cd ./apriori/apriori/src/ && make >/dev/null 2>&1)
(cd ./fpgrowth/fpgrowth/src/ && make >/dev/null 2>&1)

# Check if executables exist
if [ ! -f "$APRIORI_EXEC" ] || [ ! -f "$FPGROWTH_EXEC" ]; then
    echo "ERROR: Compilation failed. Please check the source code."
    exit 1
fi
echo "INFO: Compilation successful."

# Create results file with header
cd "$(dirname "$0")"
echo "SupportThreshold,Algorithm,RunTime(s)" > $RESULTS_FILE

echo "INFO: Starting experiments. This may take some minutes..."

for support in "${SUPPORT_THRESHOLDS[@]}"; do
    echo "INFO: Running with support threshold: $support%"

    echo "  -> Apriori..."
    RUNTIME=$( { TIMEFORMAT='%R'; time $APRIORI_EXEC -s$support $DATASET /dev/null 1>/dev/null 2>&1; } 2>&1 )
    echo "$support,Apriori,$RUNTIME" >> $RESULTS_FILE

    echo "  -> FP-Growth..."
    RUNTIME=$( { TIMEFORMAT='%R'; time $FPGROWTH_EXEC -s $support $DATASET /dev/null 1>/dev/null 2>&1; } 2>&1 )
    echo "$support,FP-Growth,$RUNTIME" >> $RESULTS_FILE
done

echo "Runtime data saved to '$RESULTS_FILE'"
