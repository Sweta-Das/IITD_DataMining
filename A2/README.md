# Assignment 2

This folder contains the code and scripts for Assignment 2 of COL761 / AIL7026.

Problem statement:

- [A2 PDF](docs/2502_COL761_A2.pdf)

## Contents

- `q1/` - KMeans clustering and elbow-method analysis
- `q2/` - forest-fire spread / edge-blocking optimization and evaluation
- `requirements.txt` - Python dependencies for this assignment

## Setup

Create a virtual environment and install the dependencies from the assignment root:

```bash
cd A2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Q1

Question 1 lives in `A2/q1/` and runs KMeans on a dataset either fetched from the course server or loaded from a local `.npy` file.

```bash
cd A2/q1
python3 Q1.py 1
python3 Q1.py dataset_1.npy
```

Behavior:

- prints the selected `k` to stdout
- writes `plot.png`
- also writes a second diagnostic plot named `dataset_<arg>_plot2.png`

Note:

- download mode requires network access and uses the student id embedded in `Q1.py`

## Q2

Question 2 lives in `A2/q2/` and solves the forest-fire edge-blocking problem.

Run the solver directly:

```bash
cd A2/q2
python3 Q2.py <graph_file> <seed_file> <out_file> <k> <r> <hops>
```

Arguments:

- `k` - number of edges to block
- `r` - number of Monte Carlo realizations
- `hops` - `-1` for unlimited propagation, otherwise a hop limit

Convenience wrapper:

```bash
cd A2/q2
bash forest_fire.sh <graph_file> <seed_file> <out_file> <k> <r> <hops>
```

Evaluation:

```bash
cd A2/q2
bash Eval/evaluate.sh <graph_file> <seed_file> <blocked_file> <k> <num_sim> [hops]
```

Included sample datasets:

- `A2/q2/dataset1/` - `k 50`, `num_sim 50`, `hops -1`
- `A2/q2/dataset2/` - `k 30`, `num_sim 50`, `hops 3`
