# Assignment 1 (A1) — COL761 / AIL7026 Data Mining

This folder contains our solutions and helper scripts for **Assignment 1**.

## Folder layout

- `docs/` — problem statement PDF
- `q1/` — frequent itemset mining (Apriori vs FP-Growth), dataset generation, plots
- `q2/` — frequent subgraph mining comparison (gSpan vs FSG vs Gaston), timing + plot
- `q3/` — graph utilities + scripts (see `q3.pdf`)

## Requirements

- `python3`
- Some scripts use common CLI tools:
  - `timeout` (Linux) or `gtimeout` (macOS via `coreutils`)
  - `bc`
- Some questions expect **external algorithm implementations/binaries** (not necessarily tracked in this repo). See the per-question usage below.

## Q1 — Frequent Itemset Mining

Location: `A1/q1/`

### Task 1.1: Run Apriori vs FP-Growth

```bash
cd A1/q1
bash q1_1.sh <apriori_exec> <fpgrowth_exec> <dataset> <out_dir>
```

Notes:
- `q1_1.sh` attempts a local `make` under `./apriori/...` and `./fpgrowth/...` if those directories exist.
- Outputs a `results.csv` in `<out_dir>` plus per-run output files.

### Task 1.2: Generate dataset + run mining

```bash
cd A1/q1
bash q1_2.sh <universal_itemset> <num_transactions>
```

This generates `generated_transactions.dat` (in `A1/q1/`) and runs `q1_1.sh` on it.

## Q2 — Frequent Subgraph Mining Comparison

Location: `A1/q2/`

Run the full pipeline (convert dataset formats → run algorithms at multiple supports → plot):

```bash
cd A1/q2
bash q2.sh <gspan_exe> <fsg_exe> <gaston_exe> <dataset> <out_dir>
```

Outputs:
- Per-support outputs like `gspan5`, `fsg10`, `gaston50`, ...
- `timing_results.json`
- `plot.png` (performance comparison)

## Q3 — Graph scripts

Location: `A1/q3/`

Set up a local venv for this question:

```bash
cd A1/q3
bash env.sh
```

Run discriminative subgraph identification:

```bash
cd A1/q3
bash identify.sh <graph_dataset> <discriminative_subgraphs_out>
```

Other helpers live in the same folder (`convert.sh`, `generate_candidates.sh`, `match.py`, etc.). Refer to `A1/q3/q3.pdf` for expected inputs/outputs.

