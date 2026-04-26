"""
Unified training entry point for COL761 A3 Q2.

Expected evaluator command:
python train.py --dataset A|B|C --task node|link --data_dir ... --model_dir ... --kerberos ...
"""

import argparse
import sys

import train_A
import train_A_appnp
import train_B
import train_C


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["A", "B", "C"])
    parser.add_argument("--task", required=True, choices=["node", "link"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--kerberos", required=True)

    # Optional common hyperparameters, so evaluator can still pass only required args.
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)

    args = parser.parse_args()

    if args.dataset in ["A", "B"] and args.task != "node":
        raise ValueError("Datasets A and B require --task node")
    if args.dataset == "C" and args.task != "link":
        raise ValueError("Dataset C requires --task link")

    # Build namespace expected by individual train scripts.
    if args.dataset == "A":
        ns = argparse.Namespace(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            kerberos=args.kerberos,
            hidden=args.hidden if args.hidden is not None else 128,
            layers=2,
            dropout=args.dropout if args.dropout is not None else 0.75,
            lr=args.lr if args.lr is not None else 0.01,
            weight_decay=args.weight_decay if args.weight_decay is not None else 5e-4,
            epochs=args.epochs if args.epochs is not None else 1000,
            patience=args.patience if args.patience is not None else 80,
        )
        # Use stronger multi-seed APPNP for Graph A.
        ns.hidden = 64
        ns.dropout = 0.5
        ns.K = 10
        ns.alpha = 0.1
        ns.lr = 0.01
        ns.weight_decay = 5e-4
        ns.epochs = 1000
        ns.patience = 80
        ns.seeds = "0,1,2,3,4,5,6,7,8,9"
        train_A_appnp.main(ns)

    elif args.dataset == "B":
        ns = argparse.Namespace(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            kerberos=args.kerberos,
            hidden=args.hidden if args.hidden is not None else 128,
            dropout=args.dropout if args.dropout is not None else 0.4,
            fan_out="10,5",
            batch_size=512,
            lr=args.lr if args.lr is not None else 0.0005,
            weight_decay=args.weight_decay if args.weight_decay is not None else 5e-4,
            epochs=args.epochs if args.epochs is not None else 10,
            patience=args.patience if args.patience is not None else 3,
        )
        train_B.main(ns)

    elif args.dataset == "C":
        # Graph C: raw normalized entity embeddings are a very strong baseline.
        # This saves a lightweight checkpoint; predict.py computes cosine scores.
        import os
        import torch

        os.makedirs(args.model_dir, exist_ok=True)
        model_path = os.path.join(args.model_dir, f"{args.kerberos}_model_C.pt")

        torch.save({
            "model_config": {
                "model_type": "PairCosineBlendLink",
                "blend_w": 0.4,
                "smooth_steps": 3,
                "smooth_beta": 0.6,
            },
            "dataset": "C",
            "task": "link",
            "note": "Blends raw cosine with graph-smoothed cosine scores.",
        }, model_path)

        print(f"Saved PairCosineBlendLink checkpoint to: {model_path}")


if __name__ == "__main__":
    main()
