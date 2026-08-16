# Assignment 1

This folder contains the code and scripts for Assignment 1 of COL761 / AIL7026.

Problem statement:

- [A1 PDF](docs/2502_COL761_A1_final.pdf)

## Contents

- `q1/` - frequent itemset mining experiments, including Apriori vs FP-Growth
- `q2/` - frequent subgraph mining comparison across gSpan, FSG, and Gaston
- `q3/` - graph utilities and discriminative subgraph identification scripts

## Requirements

- `python3`
- Standard shell tools such as `make`, `timeout` or `gtimeout`, and `bc`
- External binaries for some questions, if you are running the full pipelines locally

## Q1

Question 1 is split into two parts and lives in `A1/q1/`.

### Task 1.1

Runs Apriori and FP-Growth over a dataset at multiple support thresholds.

```bash
cd A1/q1
bash q1_1.sh <apriori_exec> <fpgrowth_exec> <dataset> <out_dir>
```

Outputs:

- `results.csv`
- per-run algorithm output files in `<out_dir>`
- `plot.png` if plotting succeeds

### Task 1.2

Generates a synthetic transaction dataset and then reuses Task 1.1.

```bash
cd A1/q1
bash q1_2.sh <universal_itemset> <num_transactions>
```

This writes `generated_transactions.dat` and then launches the comparison pipeline.

## Q2

Question 2 lives in `A1/q2/` and compares frequent subgraph mining implementations.

```bash
cd A1/q2
bash q2.sh <gspan_exe> <fsg_exe> <gaston_exe> <dataset> <out_dir>
```

The script:

- converts the input dataset into the formats expected by the three miners
- runs each miner across multiple support values
- writes timing data to `timing_results.json`
- generates `plot.png`

## Q3

Question 3 lives in `A1/q3/` and focuses on graph processing helpers and discriminative subgraph identification.

Typical setup:

```bash
cd A1/q3
bash env.sh
```

Run the main identification pipeline with:

```bash
cd A1/q3
bash identify.sh <graph_dataset> <discriminative_subgraphs_out>
```

Supporting scripts in the same folder include `convert.sh`, `convert.py`, `generate_candidates.sh`, `match.py`, and `graph_utils.py`. See [q3.pdf](q3/q3.pdf) for the expected inputs and outputs.
