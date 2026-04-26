import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from load_dataset import load_dataset
from models import APPNPNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
    val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
    return train_acc, val_acc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = load_dataset("A", args.data_dir).to(device)

    in_channels = data.x.shape[1]
    num_classes = int(data.y.max().item()) + 1

    print(
        f"Graph A APPNP | nodes={data.num_nodes} edges={data.edge_index.shape[1]} "
        f"features={in_channels} classes={num_classes}"
    )

    seeds = getattr(args, "seeds", "0,1,2,3,4,5,6,7,8,9")
    if isinstance(seeds, str):
        seeds = [int(x) for x in seeds.split(",")]

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, f"{args.kerberos}_model_A.pt")

    best_global = 0.0
    best_info = None
    start = time.time()

    for seed in seeds:
        set_seed(seed)

        model = APPNPNet(
            in_channels=in_channels,
            hidden_channels=args.hidden,
            out_channels=num_classes,
            dropout=args.dropout,
            K=args.K,
            alpha=args.alpha,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        best_seed_val = 0.0
        bad = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()

            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                train_acc, val_acc = evaluate(model, data)

                if val_acc > best_seed_val:
                    best_seed_val = val_acc
                    bad = 0
                else:
                    bad += 1

                if val_acc > best_global:
                    best_global = val_acc
                    best_info = (seed, epoch)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "model_config": {
                                "model_type": "APPNPNet",
                                "in_channels": in_channels,
                                "hidden_channels": args.hidden,
                                "out_channels": num_classes,
                                "dropout": args.dropout,
                                "K": args.K,
                                "alpha": args.alpha,
                            },
                            "dataset": "A",
                            "task": "node",
                            "seed": seed,
                            "epoch": epoch,
                            "best_val_acc": best_global,
                        },
                        model_path,
                    )

                print(
                    f"seed={seed} epoch={epoch:04d} loss={loss.item():.4f} "
                    f"train={train_acc:.4f} val={val_acc:.4f} "
                    f"best={best_global:.4f} [{time.time() - start:.0f}s]"
                )

                if bad >= args.patience:
                    print(f"seed={seed} early stop at epoch={epoch}")
                    break

        print(f"seed={seed} best_val={best_seed_val:.4f}")

    print(f"\nBest validation Accuracy: {best_global:.4f}")
    print(f"Best seed/epoch: {best_info}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--kerberos", required=True)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    main(parser.parse_args())
