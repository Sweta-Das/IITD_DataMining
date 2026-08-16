# Assignment 3

This folder contains the code and scripts for Assignment 3 of COL761 / AIL7026.

Problem statement:

- [A3 PDF](docs/COL761_H3.pdf)

## Contents

- `q1/` - high-dimensional search / representative item ranking
- `q2/` - graph neural network training and prediction for datasets A, B, and C
- `requirements.txt` - Python dependencies used by the assignment
- `q2/requirements.txt` - extra dependency notes for the GNN pipeline

## Q1

Question 1 lives in `A3/q1/Q1/`.

Key files:

- `main.py` - evaluation harness and CLI wrapper
- `submission.py` - student solution entry point
- `WALKTHROUGH.md` - notes on the runner and expected contract

The provided solution writes the submission outputs for the two datasets used in the assignment:

- `output_D1.txt`
- `output_D2.txt`

The accompanying report is in:

- `COL761_A3_Q1_Report.pdf`

## Q2

Question 2 lives in `A3/q2/` and provides training, prediction, and evaluation scripts for three datasets:

- `A` - node classification
- `B` - node classification
- `C` - link prediction

Main entry points:

- `src/train.py` - unified training CLI
- `src/predict.py` - produces prediction files for the evaluator
- `evaluate.py` - local evaluation helper
- `src/load_dataset.py` - dataset loader
- `src/models.py` - model definitions

### Typical workflow

Train a model:

```bash
cd A3/q2/src
python3 train.py --dataset A --task node --data_dir /path/to/datasets --model_dir /path/to/models --kerberos YOUR_KERBEROS
```

Generate predictions:

```bash
cd A3/q2/src
python3 predict.py --dataset A --task node --data_dir /path/to/datasets --model_dir /path/to/models --output_dir /path/to/outputs --kerberos YOUR_KERBEROS
```

Evaluate locally:

```bash
cd A3
python3 evaluate.py --dataset A --task node --data_dir /path/to/datasets --output_dir /path/to/outputs --kerberos YOUR_KERBEROS
```

Notes:

- dataset `B` uses neighbor sampling, so `pyg_lib` may be required in addition to the standard PyTorch Geometric stack
- `A3/q2/README.md` contains submission-specific notes for the exact environment used here

Setup hint:

```bash
cd A3
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r q2/requirements.txt
```
