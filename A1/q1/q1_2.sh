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