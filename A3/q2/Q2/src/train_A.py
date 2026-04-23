"""
Train GraphSAGE on Graph A – 7-class node classification (metric: Accuracy).

Usage
-----
python train_A.py \
    --data_dir  /absolute/path/to/datasets \
    --model_dir /path/to/save/models \
    --kerberos  YOUR_KERBEROS \
    [--hidden 256] [--layers 3] [--dropout 0.5] \
    [--lr 0.001] [--epochs 2000] [--patience 200]
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F

from load_dataset import load_dataset
from models import GraphSAGE


# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out  = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out  = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
    val_acc   = (pred[data.val_mask]   == data.y[data.val_mask]  ).float().mean().item()
    return train_acc, val_acc


# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────
    data = load_dataset("A", args.data_dir)
    data = data.to(device)

    num_classes  = int(data.y.max().item()) + 1
    in_channels  = data.x.shape[1]
    print(f"Graph A  |  nodes={data.num_nodes}  edges={data.edge_index.shape[1]}  "
          f"features={in_channels}  classes={num_classes}")

    # ── Model ──────────────────────────────────────────────────────────────
    model = GraphSAGE(
        in_channels     = in_channels,
        hidden_channels = args.hidden,
        out_channels    = num_classes,
        num_layers      = args.layers,
        dropout         = args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    # ── Training loop ──────────────────────────────────────────────────────
    os.makedirs(args.model_dir, exist_ok=True)
    model_path  = os.path.join(args.model_dir, f"{args.kerberos}_model_A.pt")

    best_val_acc   = 0.0
    patience_count = 0
    start          = time.time()

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, data, optimizer)
        scheduler.step()

        if epoch % 10 == 0:
            train_acc, val_acc = evaluate(model, data)

            if val_acc > best_val_acc:
                best_val_acc   = val_acc
                patience_count = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "model_type"     : "GraphSAGE",
                        "in_channels"    : in_channels,
                        "hidden_channels": args.hidden,
                        "out_channels"   : num_classes,
                        "num_layers"     : args.layers,
                        "dropout"        : args.dropout,
                    },
                    "dataset": "A",
                    "task"   : "node",
                    "epoch"  : epoch,
                    "best_val_acc": best_val_acc,
                }, model_path)
            else:
                patience_count += 1

            elapsed = time.time() - start
            print(f"Epoch {epoch:04d}  loss={loss:.4f}  "
                  f"train={train_acc:.4f}  val={val_acc:.4f}  "
                  f"best={best_val_acc:.4f}  [{elapsed:.0f}s]")

            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for "
                      f"{args.patience * 10} epochs).")
                break

        # Hard time cap (1 hour = 3600 s)
        if time.time() - start >= 3600:
            print("Reached 1-hour time limit, stopping.")
            break

    print(f"\nBest validation Accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {model_path}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",     required=True)
    parser.add_argument("--model_dir",    required=True)
    parser.add_argument("--kerberos",     required=True)
    parser.add_argument("--hidden",       type=int,   default=256)
    parser.add_argument("--layers",       type=int,   default=3)
    parser.add_argument("--dropout",      type=float, default=0.5)
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs",       type=int,   default=2000)
    parser.add_argument("--patience",     type=int,   default=200,
                        help="Early-stop after this many *checks* (×10 epochs)")
    main(parser.parse_args())
