#!/bin/bash
# Usage: bash identify.sh <graph_db> <feature_out>

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <graph_dataset> <discriminative_subgraphs>"
    exit 1
fi

source venv/bin/activate
python3 identify.py "$1" "$2"