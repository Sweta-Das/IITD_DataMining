#!/bin/bash
set -euo pipefail 

echo "INFO: Starting Task-1.1..."

# Global Time Limit
GLOBAL_LIMIT=3600
GLOBAL_START=$(date +%s)

# Ensure 'timeout' or 'gtimeout' is available
if command -v timeout >/dev/null 2>&1; then
  TO="timeout"
elif command -v gtimeout >/dev/null 2>&1; then
  TO="gtimeout"
  KILL_AFTER_TIMEOUT_SEC="5" # gtimeout has explicit -k option for aggressive killing
else
  echo "ERROR: timeout/gtimeout not found"
  exit 1
fi

# 'bc' is needed for floating-point arithmetic in runtime calculations.
if ! command -v bc &> /dev/null; then
    echo "bc required"
    exit 1
fi

# Local Compilation
echo "INFO: Compiling local source code..."
if (pushd ./apriori/apriori/src/ >/dev/null && make >/dev/null 2>&1 && popd >/dev/null); then
    echo "INFO: Apriori local compilation successful."
else
    echo "WARNING: Local compilation of Apriori failed."
fi

if (pushd ./fpgrowth/fpgrowth/src/ >/dev/null && make >/dev/null 2>&1 && popd >/dev/null); then
    echo "INFO: FP-Growth local compilation successful."
else
    echo "WARNING: Local compilation of FP-Growth failed."
fi
echo "INFO: Compilation step finished."

# Args
if [ "$#" -ne 4 ]; then
  echo "ERROR: Invalid number of arguments provided."
  echo "Usage: $0 <apriori_exec> <fpgrowth_exec> <dataset> <out_dir>"
  exit 1
fi

APRIORI_EXEC="$1"
FPGROWTH_EXEC="$2"
DATASET="$3"
OUT="$4"

# Verify Executables and Dataset Exist 
if ! [[ -f "$APRIORI_EXEC" && -x "$APRIORI_EXEC" ]]; then
    echo "CRITICAL ERROR: Apriori executable not found or not executable at '$APRIORI_EXEC'."
    exit 1
fi
if ! [[ -f "$FPGROWTH_EXEC" && -x "$FPGROWTH_EXEC" ]]; then
    echo "CRITICAL ERROR: FP-Growth executable not found or not executable at '$FPGROWTH_EXEC'."
    exit 1
fi
if ! [[ -f "$DATASET" ]]; then
    echo "CRITICAL ERROR: Dataset file not found at '$DATASET'."
    exit 1
fi
echo "INFO: All provided executables and dataset found."


mkdir -p "$OUT"

SUPPORTS=(5 10 25 50 90)
CSV="$OUT/results.csv" 

echo "SupportThreshold,Algorithm,RunTime(s),Status" > "$CSV"


# helper function (enhanced with gtimeout -k and precise runtime)

run_alg () {
  EXEC=$1
  NAME=$2
  SUPPORT=$3
  OUTFILE=$4
  LIMIT=$5

  NOW=$(date +%s)
  ELAPSED=$((NOW - GLOBAL_START))
  REMAINING=$((GLOBAL_LIMIT - ELAPSED))

  if (( REMAINING <= 0 )); then
      echo "GLOBAL LIMIT 3600s reached. Stopping cleanly."
      exit 0
  fi

  if (( LIMIT > REMAINING )); then
      LIMIT=$REMAINING
  fi

  echo "  -> $NAME ($SUPPORT%) | limit=${LIMIT}s"

  START=$(date +%s.%N) 
  : > "$OUTFILE"

  set +e   

  if [[ "$TO" == "gtimeout" ]]; then
        "$TO" -k "${KILL_AFTER_TIMEOUT_SEC}s" "${LIMIT}s" \
            "$EXEC" -s"$SUPPORT" "$DATASET" "$OUTFILE"
  else
        "$TO" "${LIMIT}s" \
            "$EXEC" -s"$SUPPORT" "$DATASET" "$OUTFILE"
  fi

  STATUS_CODE=$?
  set -e   

  END=$(date +%s.%N)
  RUNTIME=$(echo "$END - $START" | bc -l)

  STATUS="Completed"

  if [ "$STATUS_CODE" -eq 124 ]; then
    STATUS="TimedOut"
    RUNTIME="${LIMIT}.000" # Explicitly set runtime to timeout value
  elif [ "$STATUS_CODE" -eq 15 ]; then
    STATUS="NoFrequentSets"
  elif [ "$STATUS_CODE" -ne 0 ]; then
    STATUS="Error (Exit ${STATUS_CODE})"
  fi

  echo "$SUPPORT,$NAME,$RUNTIME,$STATUS" >> "$CSV"
}


# Main loop

for s in "${SUPPORTS[@]}"; do
  echo "INFO: Support = $s%"

  case $s in
    5)  APR_T=1800 ;;  
    10) APR_T=1200 ;;
    25) APR_T=600  ;;
    50) APR_T=300  ;;
    90) APR_T=120  ;;
    *) APR_T=3600 ;; 
  esac

  FP_T=900   

  run_alg "$APRIORI_EXEC" "Apriori" "$s" "$OUT/ap${s}" "$APR_T" 
  run_alg "$FPGROWTH_EXEC" "FP-Growth" "$s" "$OUT/fp${s}" "$FP_T" 

  echo ""
done


# Plot

echo "INFO: Attempting to generate plot..."
if [[ -f "generate_plot.py" ]]; then
    python generate_plot.py "$CSV" "$OUT/plot.png"
    if [[ $? -ne 0 ]]; then
        echo "ERROR: Plot generation failed. Please check 'generate_plot.py' and its dependencies."
    else
        echo "INFO: Plot generated successfully: '$OUT/plot.png'."
    fi
else
    echo "WARNING: 'generate_plot.py' not found in the current directory. Skipping plot generation."
fi

echo "INFO: Script execution completed."