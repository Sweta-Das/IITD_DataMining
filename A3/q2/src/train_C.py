"""
Train GCN encoder + MLP decoder on Graph C – link prediction (metric: Hits@50).

Graph C stats:
  nodes=3,327  features=3,703
  train_pos=3,918  train_neg=3,870
  valid_pos=227    valid_neg_hard=[227, 500, 2]  (500 hard negatives per pos)

Hits@50 evaluation:
  For each validation positive edge (u, v), we rank it against the 500
  hard negatives in  valid_neg_hard[i].  Hits@50 = fraction of positives
  with rank <= 50 among those 500 hard negatives.

Usage
-----
python train_C.py \
    --data_dir  /absolute/path/to/datasets \
    --model_dir /path/to/save/models \
    --kerberos  YOUR_KERBEROS \
    [--hidden 256] [--layers 2] [--dropout 0.3] \
    [--lr 0.001] [--epochs 1000] [--patience 100]
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F

from load_dataset import load_dataset
from models import GCNEncoder, LinkPredictor


# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(encoder, predictor, data, optimizer):
    encoder.train()
    predictor.train()
    optimizer.zero_grad()

    z = encoder(data.x, data.edge_index)

    # Positive edges from training set
    pos = data.train_pos                                       # [2, M_pos]
    pos_score = predictor(z[pos[0]], z[pos[1]])

    # Negative edges from training set (pre-sampled)
    neg = data.train_neg                                       # [2, M_neg]
    neg_score = predictor(z[neg[0]], z[neg[1]])

    # Balance positive / negative counts
    n = min(pos_score.shape[0], neg_score.shape[0])
    scores = torch.cat([pos_score[:n], neg_score[:n]])
    labels = torch.cat([
        torch.ones (n, device=scores.device),
        torch.zeros(n, device=scores.device),
    ])

    loss = F.binary_cross_entropy_with_logits(scores, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def hits_at_k(pos_scores: torch.Tensor,
              neg_hard: torch.Tensor,
              predictor: torch.nn.Module,
              z: torch.Tensor,
              K: int = 50) -> float:
    """
    Hits@K using per-positive hard negatives.

    pos_scores : [V]          raw logits for V positive val edges
    neg_hard   : [V, 500, 2]  500 hard negative node-pairs per positive
    z          : [N, d]       node embeddings
    """
    V    = pos_scores.shape[0]
    hits = 0

    pos_probs = torch.sigmoid(pos_scores)                   # [V]

    for i in range(V):
        pairs = neg_hard[i]                                 # [500, 2]
        neg_s = torch.sigmoid(
            predictor(z[pairs[:, 0]], z[pairs[:, 1]])
        )                                                   # [500]

        rank = int((neg_s >= pos_probs[i]).sum().item()) + 1
        if rank <= K:
            hits += 1

    return hits / V


@torch.no_grad()
def evaluate(encoder, predictor, data, device) -> float:
    encoder.eval()
    predictor.eval()

    z        = encoder(data.x, data.edge_index)
    val_pos  = data.val_pos.to(device)          # [2, V]
    neg_hard = data.val_neg_hard.to(device)     # [V, 500, 2]

    pos_scores = predictor(z[val_pos[0]], z[val_pos[1]])    # [V]
    return hits_at_k(pos_scores, neg_hard, predictor, z)


# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = load_dataset("C", args.data_dir)
    data = data.to(device)

    in_channels = data.x.shape[1]
    print(f"Graph C  |  nodes={data.num_nodes}  "
          f"train_pos={data.train_pos.shape[1]}  "
          f"train_neg={data.train_neg.shape[1]}  "
          f"val_pos={data.val_pos.shape[1]}  "
          f"features={in_channels}")

    # ── Models ─────────────────────────────────────────────────────────────
    encoder = GCNEncoder(
        in_channels     = in_channels,
        hidden_channels = args.hidden,
        out_channels    = args.hidden,
        num_layers      = args.layers,
        dropout         = args.dropout,
    ).to(device)

    predictor = LinkPredictor(
        in_channels     = args.hidden,
        hidden_channels = args.hidden,
        num_layers      = 2,
        dropout         = args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    # ── Training loop ──────────────────────────────────────────────────────
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir,
                              f"{args.kerberos}_model_C.pt")

    best_hits = 0.0
    pat_count = 0
    start     = time.time()

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(encoder, predictor, data, optimizer)
        scheduler.step()

        if epoch % 10 == 0:
            hits = evaluate(encoder, predictor, data, device)

            if hits > best_hits:
                best_hits = hits
                pat_count = 0
                torch.save({
                    "encoder_state_dict"  : encoder.state_dict(),
                    "predictor_state_dict": predictor.state_dict(),
                    "model_config": {
                        "in_channels"    : in_channels,
                        "hidden_channels": args.hidden,
                        "out_channels"   : args.hidden,
                        "num_layers"     : args.layers,
                        "dropout"        : args.dropout,
                    },
                    "dataset"  : "C",
                    "task"     : "link",
                    "epoch"    : epoch,
                    "best_hits": best_hits,
                }, model_path)
            else:
                pat_count += 1

            elapsed = time.time() - start
            print(f"Epoch {epoch:04d}  loss={loss:.4f}  "
                  f"val_hits@50={hits:.4f}  best={best_hits:.4f}  "
                  f"[{elapsed:.0f}s]")

            if pat_count >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        if time.time() - start >= 7200:
            print("Reached 2-hour time limit.")
            break

    print(f"\nBest val Hits@50 : {best_hits:.4f}")
    print(f"Model saved to   : {model_path}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",     required=True)
    parser.add_argument("--model_dir",    required=True)
    parser.add_argument("--kerberos",     required=True)
    parser.add_argument("--hidden",       type=int,   default=256)
    parser.add_argument("--layers",       type=int,   default=2)
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs",       type=int,   default=1000)
    parser.add_argument("--patience",     type=int,   default=100)
    main(parser.parse_args())
