# Assignment 2 (A2) — COL761 / AIL7026 Data Mining

This folder contains our solutions and helper scripts for **Assignment 2**.

## Folder layout

- `docs/` — problem statement PDF
- `requirements.txt` — Python dependencies used for A2
- `q1/` — KMeans + elbow method; saves `plot.png` and prints optimal `k`
- `q2/` — forest-fire blocking algorithm implementation + evaluator

## Setup

Create a virtual environment and install dependencies:

```bash
cd A2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Q1 — KMeans + elbow method

Location: `A2/q1/`

Run with a dataset number (downloads from the course server) **or** with a local `.npy` file:

```bash
cd A2/q1
python3 Q1.py 1
# or
python3 Q1.py dataset_1.npy
```

Notes:
- Download mode requires network access and uses a hard-coded `student_id` in `Q1.py`.

Outputs:
- Prints the selected `k` to stdout
- Writes `plot.png` in the current working directory (typically `A2/q1/`)

## Q2 — Forest fire spread (edge blocking)

Location: `A2/q2/`

### Run the solver

`Q2.py` usage:

```bash
cd A2/q2
python3 Q2.py <graph_file> <seed_file> <out_file> <k> <r> <hops>
```

- `k`: number of edges to output/block
- `r`: number of Monte-Carlo realizations
- `hops`: `-1` for unlimited, else restrict spread to that many hops

Convenience wrapper:

```bash
cd A2/q2
bash forest_fire.sh <graph_file> <seed_file> <out_file> <k> <r> <hops>
```

Included datasets:
- `A2/q2/dataset1/` (`config.txt`: `k 50`, `num_sim 50`, `hops -1`)
- `A2/q2/dataset2/` (`config.txt`: `k 30`, `num_sim 50`, `hops 3`)

### Evaluate an output

Evaluator script:

```bash
cd A2/q2
bash Eval/evaluate.sh <graph_file> <seed_file> <blocked_file> <k> <num_sim> [hops]
```

Note: the evaluator comments mention absolute paths, but relative paths also work when you run it from `A2/q2/`.

Example (dataset1):

```bash
cd A2/q2
bash Eval/evaluate.sh dataset1/dataset_1.txt dataset1/seedset_1.txt dataset1/output.txt 50 50 -1
```
