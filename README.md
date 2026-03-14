# IIT Delhi — Data Mining (COL761 / AIL7026): Assignments

This repository contains my coursework for **Data Mining (COL761 / AIL7026)** at **IIT Delhi**, organized by assignment/homework.

> Note: This is **not** an official course repository. If you’re taking the course, please follow the course’s collaboration / academic integrity policy—do not copy solutions.

## Repository layout

- `HW0/` — Homework 0 (setup/proofs and submission artifacts).
- `A1/` — Assignment 1
  - `A1/docs/` — problem statement PDF
  - `A1/q1/` — frequent itemset mining experiments (Apriori vs FP-Growth), dataset generation, plots
  - `A1/q2/` — frequent subgraph mining comparison pipeline (gSpan, FSG, Gaston), timing + plot
  - `A1/q3/` — graph utilities + scripts (see `A1/q3/q3.pdf`)
- `A2/` — Assignment 2
  - `A2/docs/` — problem statement PDF
  - `A2/q1/` — KMeans + elbow method; saves `plot.png` and prints optimal `k`
  - `A2/q2/` — graph algorithm solution (run with CLI args; see usage below)
  - `A2/requirements.txt` — Python dependencies used for A2

## Setup

Most code is Python and intended to run on macOS/Linux with `python3`.

Create a virtual environment and install dependencies (A2):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r A[num]/requirements.txt
```

Some A1 scripts also rely on common command-line tools:

- `make` (to compile external implementations, if you use them)
- `timeout` (Linux) or `gtimeout` (macOS via coreutils)
- `bc`


## Outputs and tracked files

This repo intentionally does **not** commit some large/generated artifacts and some third-party code directories. See `.gitignore` for details.
