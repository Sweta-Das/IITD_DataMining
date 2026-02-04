#!/bin/bash

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install requirements
# numpy is critical for the vector operations
# networkx is used for feature extraction
pip install numpy networkx tqdm