"""
Dataset loader for COL761 A3 Q2.

Usage
-----
python load_dataset.py --dataset A|B|C --data_dir /absolute/path/to/datasets

Actual data layout (discovered from downloaded files)
------------------------------------------------------
A / B  (node classification)
    <data_dir>/<dataset>/data.pt   →  PyG Data with:
        x              [N, F]   all node features
        edge_index     [2, E]   edges (whole graph)
        y              [L]      labels for labeled nodes only
        labeled_nodes  [L]      node indices that have a label
        train_mask     [L]      bool – training subset of labeled_nodes
        val_mask       [L]      bool – validation subset of labeled_nodes

    We expand y/masks to full-graph tensors of size N so training code
    can simply write  out[data.train_mask]  without worrying about indexing.

C  (link prediction)
    <data_dir>/C/gnn_feature          torch dict {'entity_embedding': [N, F]}
    <data_dir>/C/train_pos.txt        tab-separated node pairs
    <data_dir>/C/train_neg.txt
    <data_dir>/C/valid_pos.txt
    <data_dir>/C/valid_neg.txt
    <data_dir>/C/valid_neg_hard.npy   shape [num_val_pos, 500, 2]
"""

import argparse
import os

import numpy as np
import torch
from torch_geometric.data import Data


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pyg_mapping(raw) -> dict:
    """
    Extract the internal tensor dict from a PyG Data object.
    Works even when torch_geometric is not fully imported (uses _store).
    """
    try:
        # Normal PyG Data
        return dict(raw)
    except Exception:
        pass
    # Fallback: reach into the internal store mapping
    store = raw.__dict__.get("_store")
    if store is not None:
        return store.__dict__.get("_mapping", {})
    raise RuntimeError("Cannot extract tensor mapping from loaded data.pt")


def _read_edges(path: str) -> torch.Tensor:
    """Tab-separated edge list  →  [2, E] LongTensor."""
    edges = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                a, b = line.split("\t")
                edges.append((int(a), int(b)))
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Node classification (A and B)
# ─────────────────────────────────────────────────────────────────────────────

def _load_node(dataset: str, data_dir: str) -> Data:
    path = os.path.join(data_dir, dataset, "data.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"[load_dataset] '{path}' not found.\n"
            f"Download dataset {dataset} (kerberos login) and place data.pt "
            f"under '{os.path.join(data_dir, dataset)}/'."
        )

    raw = torch.load(path, weights_only=False)
    m   = _pyg_mapping(raw)

    x           = m["x"]               # [N, F]
    edge_index  = m["edge_index"]       # [2, E]
    y_partial   = m["y"]                # [L]
    labeled     = m["labeled_nodes"]    # [L]  indices into [0, N)
    train_sub   = m["train_mask"]       # [L]  bool
    val_sub     = m["val_mask"]         # [L]  bool

    N = x.shape[0]

    # ── Expand to full-graph size [N] ──────────────────────────────────────
    if y_partial.is_floating_point():
        full_y = torch.full((N,), -1.0, dtype=torch.float32)
    else:
        full_y = torch.full((N,), -1, dtype=torch.long)
    full_y[labeled] = y_partial

    full_train = torch.zeros(N, dtype=torch.bool)
    full_val   = torch.zeros(N, dtype=torch.bool)
    full_train[labeled[train_sub]] = True
    full_val  [labeled[val_sub  ]] = True

    return Data(
        x                  = x,
        edge_index         = edge_index,
        y                  = full_y,       # [N], -1 for unlabeled
        train_mask         = full_train,   # [N]
        val_mask           = full_val,     # [N]
        labeled_nodes      = labeled,      # [L]  kept for reference
        y_partial          = y_partial,    # [L]  original labels
        train_mask_partial = train_sub,    # [L]  original mask
        val_mask_partial   = val_sub,      # [L]  original mask
    )


# ─────────────────────────────────────────────────────────────────────────────
# Link prediction (C)
# ─────────────────────────────────────────────────────────────────────────────

def _load_link(data_dir: str) -> Data:
    base = os.path.join(data_dir, "C")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"[load_dataset] '{base}' not found.")

    # Node features
    feat = torch.load(os.path.join(base, "gnn_feature"), weights_only=False)
    x = feat["entity_embedding"].float()   # [N, F]

    # Edges
    train_pos = _read_edges(os.path.join(base, "train_pos.txt"))  # [2, M_pos]
    train_neg = _read_edges(os.path.join(base, "train_neg.txt"))  # [2, M_neg]
    val_pos   = _read_edges(os.path.join(base, "valid_pos.txt"))  # [2, V]
    val_neg   = _read_edges(os.path.join(base, "valid_neg.txt"))  # [2, V_neg]

    # Hard negatives for Hits@50: [V, 500, 2]
    val_neg_hard = torch.from_numpy(
        np.load(os.path.join(base, "valid_neg_hard.npy"))
    ).long()

    # Message-passing graph = positive training edges (undirected)
    edge_index = torch.cat([train_pos, train_pos[[1, 0]]], dim=1)

    return Data(
        x             = x,
        edge_index    = edge_index,
        train_pos     = train_pos,        # [2, M_pos]
        train_neg     = train_neg,        # [2, M_neg]
        val_pos       = val_pos,          # [2, V]
        val_neg       = val_neg,          # [2, V_neg]
        val_neg_hard  = val_neg_hard,     # [V, 500, 2]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(dataset: str, data_dir: str) -> Data:
    """
    Load graph dataset A, B, or C.

    Parameters
    ----------
    dataset  : "A", "B", or "C"
    data_dir : absolute path to the datasets directory
               (should contain sub-directories A/, B/, C/)

    Returns
    -------
    torch_geometric.data.Data
    """
    dataset = dataset.upper()
    if dataset in ("A", "B"):
        return _load_node(dataset, data_dir)
    elif dataset == "C":
        return _load_link(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Load COL761 A3 datasets")
    parser.add_argument("--dataset",  required=True, choices=["A", "B", "C"])
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    print(f"Loading dataset {args.dataset} from {args.data_dir} …")
    data = load_dataset(args.dataset, args.data_dir)
    print(data)
    print(f"  num_nodes  : {data.num_nodes}")
    print(f"  edge_index : {data.edge_index.shape}")
    if hasattr(data, "train_mask"):
        print(f"  train      : {data.train_mask.sum().item():,}")
        print(f"  val        : {data.val_mask.sum().item():,}")
    if hasattr(data, "train_pos"):
        print(f"  train_pos  : {data.train_pos.shape}")
        print(f"  val_pos    : {data.val_pos.shape}")
        print(f"  val_neg_hard: {data.val_neg_hard.shape}")


if __name__ == "__main__":
    main()
