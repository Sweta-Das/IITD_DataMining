"""
Generate and save predictions from a trained model.

Usage
-----
# Node classification  (A or B)
python predict.py \
    --dataset A --task node \
    --data_dir  /absolute/path/to/datasets \
    --model_dir /path/to/models \
    --output_dir /path/to/outputs \
    --kerberos  YOUR_KERBEROS

# Link prediction  (C)
python predict.py \
    --dataset C --task link \
    --data_dir  /absolute/path/to/datasets \
    --model_dir /path/to/models \
    --output_dir /path/to/outputs \
    --kerberos  YOUR_KERBEROS

Output files
------------
<output_dir>/<kerberos>_predictions_<dataset>.pt

  Node A :  { 'pred': Tensor [N, 7]  softmax probabilities }
  Node B :  { 'pred': Tensor [N]     sigmoid probability of class 1 }
  Link C :  { 'pos_scores': Tensor [V],
              'neg_scores':  Tensor [V, 500] }
"""

import argparse
import os

import torch
from torch_geometric.loader import NeighborLoader

from load_dataset import load_dataset
from models import GAT, GCNEncoder, GraphSAGE, LinkPredictor


# ──────────────────────────────────────────────────────────────────────────────

def _build_model(cfg, device):
    mtype = cfg["model_type"]
    if mtype == "GraphSAGE":
        return GraphSAGE(
            in_channels     = cfg["in_channels"],
            hidden_channels = cfg["hidden_channels"],
            out_channels    = cfg["out_channels"],
            num_layers      = cfg.get("num_layers", 3),
            dropout         = cfg.get("dropout", 0.0),   # no dropout at inference
        ).to(device)
    elif mtype == "GAT":
        return GAT(
            in_channels     = cfg["in_channels"],
            hidden_channels = cfg["hidden_channels"],
            out_channels    = cfg["out_channels"],
            num_layers      = cfg.get("num_layers", 3),
            heads           = cfg.get("heads", 4),
            dropout         = cfg.get("dropout", 0.0),
        ).to(device)
    raise ValueError(f"Unknown model_type: {mtype!r}")


# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_node_A(ckpt, data, device):
    """Full-batch inference for Graph A (2,708 nodes – fits in memory)."""
    model = _build_model(ckpt["model_config"], device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    data = data.to(device)
    logits = model(data.x, data.edge_index)        # [N, 7]
    return torch.softmax(logits, dim=1).cpu()      # [N, 7]


@torch.no_grad()
def predict_node_B(ckpt, data, device, batch_size=4096, fan_out=None):
    """
    Mini-batch inference for Graph B (2.89 M nodes).
    Runs a NeighborLoader over *all labeled nodes* (train + val).
    """
    if fan_out is None:
        cfg      = ckpt["model_config"]
        n_layers = cfg.get("num_layers", 3)
        fan_out  = [-1] * n_layers      # full neighborhood at inference

    model = _build_model(ckpt["model_config"], device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Collect predictions for all labeled nodes
    labeled = data.labeled_nodes                  # [L]
    loader  = NeighborLoader(
        data,
        num_neighbors=fan_out,
        input_nodes=labeled,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_probs = torch.zeros(data.num_nodes, dtype=torch.float32)

    for batch in loader:
        batch = batch.to(device)
        out   = model(batch.x, batch.edge_index).squeeze(-1)   # (batch_sz,)
        probs = torch.sigmoid(out[:batch.batch_size]).cpu()

        # Map back to original node indices
        orig_ids = batch.n_id[:batch.batch_size].cpu()
        all_probs[orig_ids] = probs

    return all_probs   # [N], zero for non-labeled nodes


# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_link_C(ckpt, data, device):
    """
    Predict scores for validation edges in Graph C.

    Returns
    -------
    pos_scores  : [V]         scores for positive val edges
    neg_scores  : [V, 500]    scores for hard negative val edges (per positive)
    """
    cfg = ckpt["model_config"]
    encoder = GCNEncoder(
        in_channels     = cfg["in_channels"],
        hidden_channels = cfg["hidden_channels"],
        out_channels    = cfg["out_channels"],
        num_layers      = cfg.get("num_layers", 2),
        dropout         = 0.0,
    ).to(device)

    predictor = LinkPredictor(
        in_channels     = cfg["out_channels"],
        hidden_channels = cfg["hidden_channels"],
        num_layers      = 2,
        dropout         = 0.0,
    ).to(device)

    encoder.load_state_dict(ckpt["encoder_state_dict"])
    predictor.load_state_dict(ckpt["predictor_state_dict"])
    encoder.eval()
    predictor.eval()

    data     = data.to(device)
    z        = encoder(data.x, data.edge_index)

    val_pos  = data.val_pos                        # [2, V]
    neg_hard = data.val_neg_hard                   # [V, 500, 2]
    V        = val_pos.shape[1]

    pos_scores = predictor(z[val_pos[0]], z[val_pos[1]]).cpu()   # [V]

    # Neg scores: [V, 500]
    neg_scores_list = []
    for i in range(V):
        pairs = neg_hard[i]                        # [500, 2]
        s = predictor(z[pairs[:, 0]], z[pairs[:, 1]]).cpu()      # [500]
        neg_scores_list.append(s)
    neg_scores = torch.stack(neg_scores_list, dim=0)             # [V, 500]

    return pos_scores, neg_scores


# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(args.model_dir,
                              f"{args.kerberos}_model_{args.dataset}.pt")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint: {model_path}")

    data = load_dataset(args.dataset, args.data_dir)
    print(f"Loaded dataset {args.dataset}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir,
                            f"{args.kerberos}_predictions_{args.dataset}.pt")

    if args.task == "node":
        if args.dataset == "A":
            pred = predict_node_A(ckpt, data, device)
            torch.save({"pred": pred}, out_path)
            print(f"Saved predictions shape={tuple(pred.shape)}  →  {out_path}")

        elif args.dataset == "B":
            pred = predict_node_B(ckpt, data, device)
            torch.save({"pred": pred}, out_path)
            print(f"Saved predictions shape={tuple(pred.shape)}  →  {out_path}")

        else:
            raise ValueError("--dataset must be A or B for --task node")

    elif args.task == "link":
        if args.dataset != "C":
            raise ValueError("--dataset must be C for --task link")

        pos_s, neg_s = predict_link_C(ckpt, data, device)
        torch.save({"pos_scores": pos_s, "neg_scores": neg_s}, out_path)
        print(f"Saved link predictions  pos={pos_s.shape}  neg={neg_s.shape}"
              f"  →  {out_path}")

    else:
        raise ValueError(f"Unknown --task: {args.task!r}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    required=True, choices=["A", "B", "C"])
    parser.add_argument("--task",       required=True, choices=["node", "link"])
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--model_dir",  required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--kerberos",   required=True)
    main(parser.parse_args())
