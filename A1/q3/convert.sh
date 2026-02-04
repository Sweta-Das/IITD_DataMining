#!/bin/bash
# Usage: bash convert.sh <graphs> <features> <output.npy>

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <graphs> <features> <output_npy>"
    exit 1
fi

source venv/bin/activate
python3 convert.py "$1" "$2" "$3"