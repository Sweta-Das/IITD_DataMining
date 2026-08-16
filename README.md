# IIT Delhi Data Mining Assignments

This repository contains coursework for **COL761 / AIL7026 Data Mining** at **IIT Delhi**, organized by assignment.

This is not an official course repository. If you are taking the course, follow the course collaboration and academic integrity policy.

## Layout

- `A0/` - homework 0 proofs and setup artifacts
- `A1/` - Assignment 1
- `A2/` - Assignment 2
- `A3/` - Assignment 3

Each assignment folder has its own README with the exact scripts, inputs, and outputs used there:

- [A1 README](A1/README.md)
- [A2 README](A2/README.md)
- [A3 README](A3/README.md)

The assignment PDFs are stored under:

- `A1/docs/2502_COL761_A1_final.pdf`
- `A2/docs/2502_COL761_A2.pdf`
- `A3/docs/COL761_H3.pdf`

## Setup

Most of the code is Python and is intended to run on macOS or Linux with `python3`.

For the Python-based assignments, create a virtual environment and install the relevant dependencies from the assignment folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r A2/requirements.txt
pip install -r A3/requirements.txt
```

Some scripts in `A1/` also expect standard command-line tools:

- `make` for compiling external implementations, if you are using them
- `timeout` on Linux or `gtimeout` on macOS via coreutils
- `bc` for floating-point arithmetic in shell scripts

## Notes

- Several directories contain generated outputs, plots, or submission artifacts that are kept in the repo because they are useful for grading and reproduction.
- Some external executables or datasets are not included here and must be supplied separately.
