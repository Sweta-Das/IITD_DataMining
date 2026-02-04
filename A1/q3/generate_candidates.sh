#!/bin/bash
# Usage: bash generate_candidates.sh <db_vectors.npy> <query_vectors.npy> <output.txt>

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <db_vectors> <query_vectors> <output_file>"
    exit 1
fi

source venv/bin/activate
python3 match.py "$1" "$2" "$3"