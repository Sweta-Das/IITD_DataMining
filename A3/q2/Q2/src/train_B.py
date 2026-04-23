"""
Train GraphSAGE on Graph B – binary node classification (metric: ROC-AUC).

Graph B stats:
  nodes=2,890,331  edges=24,754,822  features=1179
  labeled=578,066  train=289,033  val=289,033
  class distribution: {0: 287,132 , 1: 1,901}  (≈150:1 imbalance)

Full-batch training is infeasible (feature matrix alone ≈13.6 GB).
We use NeighborLoader (mini-batch neighbor sampling) with GraphSAGE.

Usage
-----
python train_B.py \
    --data_dir  /absolute/path/to/datasets \
    --model_dir /path/to/save/models \
    --kerberos  YOUR_KERBEROS \
    [--hidden 256] [--layers 3] [--dropout 0.3] \
    [--fan_out 15,10,5] [--batch_size 1024] \
    [--lr 0.001] [--epochs 50] [--patience 10]
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import NeighborLoader

from load_dataset import load_dataset
from models import GraphSAGE


# ──────────────────────────────────────────────────────────────────────────────

def compute_pos_weight(data: object, device: torch.device) -> torch.Tensor:
    """pos_weight = #negatives / #positives  (for BCEWithLogitsLoss)."""
    y_train = data.y[data.train_mask].float()
    n_pos   = max(y_train.sum().item(), 1)
    n_neg   = (y_train == 0).sum().item()
    return torch.tensor(n_neg / n_pos, dtype=torch.float32, device=device)


def make_loaders(data, train_nodes, val_nodes, fan_out, batch_size):
    """Create NeighborLoader instances for train and validation."""
    train_loader = NeighborLoader(
        data,
        num_neighbors=fan_out,
        input_nodes=train_nodes,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = NeighborLoader(
        data,
        num_neighbors=fan_out,
        input_nodes=val_nodes,
        batch_size=batch_size * 4,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index).squeeze(-1)  # (batch_size_full,)

        # Only compute loss on the "seed" nodes (first batch_size entries)
        n_seed = batch.batch_size
        out_seed  = out[:n_seed]
        y_seed    = batch.y[:n_seed].float()

        # Skip batch if all labels are unknown (-1)
        valid = y_seed >= 0
        if valid.sum() == 0:
            continue

        loss = criterion(out_seed[valid], y_seed[valid])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []

    for batch in loader:
        batch  = batch.to(device)
        out    = model(batch.x, batch.edge_index).squeeze(-1)
        n_seed = batch.batch_size

        prob = torch.sigmoid(out[:n_seed]).cpu().numpy()
        y    = batch.y[:n_seed].cpu().numpy()

        valid = y >= 0
        if valid.any():
            preds.append(prob[valid])
            labels.append(y[valid])

    if not preds:
        return 0.0

    preds  = np.concatenate(preds)
    labels = np.concatenate(labels)

    # Guard against single-class batches
    if len(np.unique(labels)) < 2:
        return 0.5

    return roc_auc_score(labels, preds)


# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Load & inspect ─────────────────────────────────────────────────────
    print("Loading dataset B …")
    data = load_dataset("B", args.data_dir)

    in_channels = data.x.shape[1]
    n_train     = data.train_mask.sum().item()
    n_val       = data.val_mask.sum().item()
    print(f"Graph B  |  nodes={data.num_nodes:,}  edges={data.edge_index.shape[1]:,}  "
          f"features={in_channels}  train={n_train:,}  val={n_val:,}")

    # Seed node indices (on CPU; NeighborLoader needs CPU indices)
    train_nodes = data.train_mask.nonzero(as_tuple=True)[0]
    val_nodes   = data.val_mask.nonzero(as_tuple=True)[0]

    # Fan-out per layer
    fan_out = [int(x) for x in args.fan_out.split(",")]
    n_layers = len(fan_out)

    # ── Model ──────────────────────────────────────────────────────────────
    model = GraphSAGE(
        in_channels     = in_channels,
        hidden_channels = args.hidden,
        out_channels    = 1,            # single logit  →  sigmoid for AUC
        num_layers      = n_layers,
        dropout         = args.dropout,
    ).to(device)

    # Pos-weight from full train split (on CPU)
    pos_w     = compute_pos_weight(data, torch.device("cpu"))
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_w.to(device))

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5)

    # ── Data loaders ───────────────────────────────────────────────────────
    train_loader, val_loader = make_loaders(
        data, train_nodes, val_nodes, fan_out, args.batch_size)

    # ── Training loop ──────────────────────────────────────────────────────
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir,
                              f"{args.kerberos}_model_B.pt")

    best_auc   = 0.0
    pat_count  = 0
    start      = time.time()

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        val_auc = evaluate(model, val_loader, device)

        if val_auc > best_auc:
            best_auc  = val_auc
            pat_count = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "model_type"     : "GraphSAGE",
                    "in_channels"    : in_channels,
                    "hidden_channels": args.hidden,
                    "out_channels"   : 1,
                    "num_layers"     : n_layers,
                    "dropout"        : args.dropout,
                },
                "dataset" : "B",
                "task"    : "node",
                "epoch"   : epoch,
                "best_auc": best_auc,
            }, model_path)
        else:
            pat_count += 1

        elapsed = time.time() - start
        print(f"Epoch {epoch:03d}  loss={loss:.4f}  "
              f"val_auc={val_auc:.4f}  best={best_auc:.4f}  [{elapsed:.0f}s]")

        if pat_count >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

        if time.time() - start >= 7200:
            print("Reached 2-hour time limit.")
            break

    print(f"\nBest val ROC-AUC : {best_auc:.4f}")
    print(f"Model saved to   : {model_path}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",     required=True)
    parser.add_argument("--model_dir",    required=True)
    parser.add_argument("--kerberos",     required=True)
    parser.add_argument("--hidden",       type=int,   default=256)
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--fan_out",      type=str,   default="15,10,5",
                        help="Neighbor fan-out per layer, comma-separated")
    parser.add_argument("--batch_size",   type=int,   default=1024)
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--patience",     type=int,   default=10)
    main(parser.parse_args())
