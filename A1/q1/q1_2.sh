#!/bin/bash
set -euo pipefail

echo "INFO: Starting Task-1.2 pipeline (Dataset Creation)..."

if [ "$#" -ne 2 ]; then
    echo "ERROR: Usage: $0 <universal_itemset> <num_transactions>"
    exit 1
fi

ITEM_UNIVERSE="$1"
NUM_TRANSACTIONS="$2"
OUTPUT_DATASET_NAME="generated_transactions.dat"


echo "INFO: Item Universe: $ITEM_UNIVERSE"
echo "INFO: Transactions: $NUM_TRANSACTIONS"

python3 create_dataset.py "$ITEM_UNIVERSE" "$NUM_TRANSACTIONS" "$OUTPUT_DATASET_NAME"


if [[ ! -f "$OUTPUT_DATASET_NAME" ]]; then
    echo "CRITICAL ERROR: Output file '$OUTPUT_DATASET_NAME' not created."
    exit 1
fi

echo "INFO: Dataset '$OUTPUT_DATASET_NAME' created successfully."

APRIORI_PATH="./apriori/apriori/src/apriori"
FP_PATH="./fpgrowth/fpgrowth/src/fpgrowth"
OUT_DIR="./output_task1_2"

echo "INFO: Running mining algorithms on generated dataset..."
bash q1_1.sh "$APRIORI_PATH" "$FP_PATH" "$OUTPUT_DATASET_NAME" "$OUT_DIR"

echo "INFO: Task 1.2 Pipeline Complete."
