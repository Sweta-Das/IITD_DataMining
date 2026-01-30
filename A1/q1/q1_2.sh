#!/bin/bash

# Test for args
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <universal_itemset> <num_transactions>"
    exit 1
fi

ITEM_UNIVERSE="$1"
NUM_TRANSACTIONS="$2"

echo "INFO: Starting Task-1.2..."
echo "Item Universe: $ITEM_UNIVERSE"
echo "Number of Transactions: $NUM_TRANSACTIONS"

python create_dataset.py "$ITEM_UNIVERSE" "$NUM_TRANSACTIONS"

echo "INFO: Task-1.2 completed. Dataset 'dataset.dat' created."