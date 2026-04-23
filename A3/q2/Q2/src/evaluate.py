"""
Evaluate saved predictions against ground-truth labels.

Usage
-----
python evaluate.py \
    --dataset A|B|C  --task node|link \
    --data_dir   /absolute/path/to/datasets \
    --output_dir /path/to/outputs \
    --kerberos   YOUR_KERBEROS

Metrics
-------
Graph A  (node, 7-class)  →  Accuracy
Graph B  (node, 2-class)  →  ROC-AUC
Graph C  (link)           →  Hits@50
"""

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from load_dataset import load_dataset


# ──────────────────────────────────────────────────────────────────────────────

def hits_at_k(pos_scores: torch.Tensor,
              neg_scores: torch.Tensor,
              K: int = 50) -> float:
    """
    Hits@K with per-positive hard negatives.

    pos_scores : [V]        raw logit / sigmoid score per positive edge
    neg_scores : [V, 500]   scores for 500 hard negatives per positive
    """
    pos_p  = torch.sigmoid(pos_scores).numpy()
    neg_p  = torch.sigmoid(neg_scores).numpy()   # [V, 500]
    V      = len(pos_p)
    hits   = 0

    for i in range(V):
        rank = int((neg_p[i] >= pos_p[i]).sum()) + 1
        if rank <= K:
            hits += 1

    return hits / V


# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    # ── Load saved predictions ──────────────────────────────────────────────
    pred_path = os.path.join(args.output_dir,
                             f"{args.kerberos}_predictions_{args.dataset}.pt")
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(
            f"Predictions file not found: {pred_path}\n"
            "Run predict.py first."
        )
    pred_data = torch.load(pred_path, weights_only=False)
    print(f"Loaded predictions: {pred_path}")

    # ── Load dataset for ground-truth labels ───────────────────────────────
    data = load_dataset(args.dataset, args.data_dir)

    # ── Compute metric ─────────────────────────────────────────────────────
    if args.task == "node":
        pred = pred_data["pred"]   # [N, 7] or [N]

        if args.dataset == "A":
            # Accuracy on validation nodes
            val_mask = data.val_mask
            y_true   = data.y[val_mask].numpy()

            if pred.dim() == 2:
                y_pred = pred[val_mask].argmax(dim=1).numpy()
            else:
                y_pred = (pred[val_mask] >= 0.5).long().numpy()

            acc = (y_pred == y_true).mean()
            print(f"[Dataset A]  Val Accuracy  = {acc:.4f}")
            return float(acc)

        elif args.dataset == "B":
            # ROC-AUC on validation nodes
            val_mask = data.val_mask
            y_true   = data.y[val_mask].numpy().astype(int)
            y_scores = pred[val_mask].numpy()

            # Guard: need both classes present
            if len(np.unique(y_true)) < 2:
                print("[Dataset B]  Only one class in val set – AUC undefined")
                return 0.5

            auc = roc_auc_score(y_true, y_scores)
            print(f"[Dataset B]  Val ROC-AUC   = {auc:.4f}")
            return float(auc)

        else:
            raise ValueError(f"--dataset {args.dataset} not valid for node task")

    elif args.task == "link":
        if args.dataset != "C":
            raise ValueError("--dataset must be C for link task")

        pos_scores = pred_data["pos_scores"]   # [V]
        neg_scores = pred_data["neg_scores"]   # [V, 500]

        result = hits_at_k(pos_scores, neg_scores, K=50)
        print(f"[Dataset C]  Val Hits@50   = {result:.4f}")
        return float(result)

    else:
        raise ValueError(f"Unknown --task: {args.task!r}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    required=True, choices=["A", "B", "C"])
    parser.add_argument("--task",       required=True, choices=["node", "link"])
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--kerberos",   required=True)
    main(parser.parse_args())
