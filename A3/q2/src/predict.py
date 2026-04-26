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
from models import GAT, GCNEncoder, GraphSAGE, LinkPredictor, APPNPNet


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
    elif mtype == "APPNPNet":
        return APPNPNet(
            in_channels     = cfg["in_channels"],
            hidden_channels = cfg["hidden_channels"],
            out_channels    = cfg["out_channels"],
            dropout         = cfg.get("dropout", 0.0),
            K               = cfg.get("K", 10),
            alpha           = cfg.get("alpha", 0.1),
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
        # Avoid full-neighborhood inference on huge Graph B.
        # Use bounded neighbor sampling for memory-safe prediction.
        fan_out  = [10, 5] if n_layers == 2 else [10] * n_layers

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



    if cfg.get("model_type") == "PairCosineBlendLink":
        data = data.to(device)

        x0 = data.x.float()
        x0 = torch.nn.functional.normalize(x0, p=2, dim=1)

        row, col = data.edge_index
        deg = torch.bincount(row, minlength=data.num_nodes).float().to(device).clamp_min(1)

        steps = int(cfg.get("smooth_steps", 3))
        beta = float(cfg.get("smooth_beta", 0.6))
        w = float(cfg.get("blend_w", 0.4))

        z = x0.clone()
        neigh = None
        for _ in range(steps):
            agg = torch.zeros_like(z)
            agg.index_add_(0, row, z[col])
            neigh = agg / deg.unsqueeze(1)
            z = torch.nn.functional.normalize(0.5 * x0 + 0.5 * neigh, p=2, dim=1)

        z_smooth = torch.nn.functional.normalize((1 - beta) * x0 + beta * neigh, p=2, dim=1)

        val_pos = data.val_pos
        neg_hard = data.val_neg_hard
        V = val_pos.shape[1]

        pairs_all = []
        for i in range(V):
            pos_pair = val_pos[:, i].view(1, 2)
            pairs = torch.cat([pos_pair, neg_hard[i]], dim=0)
            pairs_all.append(pairs)

        pairs_all = torch.stack(pairs_all, dim=0)  # [V, 501, 2]

        # raw cosine score matrix
        a0 = x0[pairs_all[:, :, 0]]
        b0 = x0[pairs_all[:, :, 1]]
        raw_cos = (a0 * b0).sum(dim=-1)

        # smoothed cosine score matrix
        a1 = z_smooth[pairs_all[:, :, 0]]
        b1 = z_smooth[pairs_all[:, :, 1]]
        smooth_cos = (a1 * b1).sum(dim=-1)

        # IMPORTANT: global standardization over all validation pairs,
        # matching the search code that gave 0.7753.
        raw_cos = (raw_cos - raw_cos.mean()) / (raw_cos.std() + 1e-12)
        smooth_cos = (smooth_cos - smooth_cos.mean()) / (smooth_cos.std() + 1e-12)

        scores = w * raw_cos + (1 - w) * smooth_cos

        pos_scores = scores[:, 0].cpu()
        neg_scores = scores[:, 1:].cpu()
        return pos_scores, neg_scores


    if cfg.get("model_type") == "BlendedSmoothLink":
        data = data.to(device)

        x0 = data.x.float()
        x0 = torch.nn.functional.normalize(x0, p=2, dim=1)

        row, col = data.edge_index
        deg = torch.bincount(row, minlength=data.num_nodes).float().to(device).clamp_min(1)

        def smooth_embeddings(steps, beta):
            z = x0.clone()
            neigh = None
            for _ in range(steps):
                agg = torch.zeros_like(z)
                agg.index_add_(0, row, z[col])
                neigh = agg / deg.unsqueeze(1)
                z = torch.nn.functional.normalize(0.5 * x0 + 0.5 * neigh, p=2, dim=1)
            return torch.nn.functional.normalize((1 - beta) * x0 + beta * neigh, p=2, dim=1)

        z1 = smooth_embeddings(
            int(cfg.get("s1_steps", 1)),
            float(cfg.get("s1_beta", 0.15)),
        )
        z2 = smooth_embeddings(
            int(cfg.get("s2_steps", 5)),
            float(cfg.get("s2_beta", 0.35)),
        )

        w = float(cfg.get("blend_w", 0.2))

        val_pos = data.val_pos
        neg_hard = data.val_neg_hard
        V = val_pos.shape[1]

        pos_scores_list = []
        neg_scores_list = []

        for i in range(V):
            pos_pair = val_pos[:, i].view(1, 2)
            pairs = torch.cat([pos_pair, neg_hard[i]], dim=0)

            a1 = z1[pairs[:, 0]]
            b1 = z1[pairs[:, 1]]
            s1 = (a1 * b1).sum(dim=1)

            a2 = z2[pairs[:, 0]]
            b2 = z2[pairs[:, 1]]
            s2 = -((a2 - b2) ** 2).sum(dim=1).sqrt()

            # Same-style standardization within each positive-vs-hard-negative group.
            s1 = (s1 - s1.mean()) / (s1.std() + 1e-12)
            s2 = (s2 - s2.mean()) / (s2.std() + 1e-12)

            s = w * s1 + (1 - w) * s2

            pos_scores_list.append(s[0].cpu())
            neg_scores_list.append(s[1:].cpu())

        pos_scores = torch.stack(pos_scores_list, dim=0)
        neg_scores = torch.stack(neg_scores_list, dim=0)
        return pos_scores, neg_scores


    # Strong feature-similarity baselines for Graph C.
    # RawCosineLink: cosine over provided entity embeddings.
    # SmoothedCosineLink: graph-smoothed normalized embeddings, then cosine.
    if cfg.get("model_type") in ["RawCosineLink", "SmoothedCosineLink"]:
        data = data.to(device)
        x0 = data.x.float()
        x0 = x0 / (x0.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12))

        x = x0

        if cfg.get("model_type") == "SmoothedCosineLink":
            steps = int(cfg.get("smooth_steps", 5))
            beta = float(cfg.get("beta", 0.25))

            row, col = data.edge_index
            deg = torch.bincount(row, minlength=data.num_nodes).float().to(device).clamp_min(1)

            z = x0.clone()
            neigh = None
            for _ in range(steps):
                agg = torch.zeros_like(z)
                agg.index_add_(0, row, z[col])
                neigh = agg / deg.unsqueeze(1)
                z = torch.nn.functional.normalize(0.5 * x0 + 0.5 * neigh, p=2, dim=1)

            x = torch.nn.functional.normalize((1 - beta) * x0 + beta * neigh, p=2, dim=1)

        val_pos = data.val_pos
        neg_hard = data.val_neg_hard
        V = val_pos.shape[1]

        pos_scores = (x[val_pos[0]] * x[val_pos[1]]).sum(dim=1).cpu()

        neg_scores_list = []
        for i in range(V):
            pairs = neg_hard[i]
            s = (x[pairs[:, 0]] * x[pairs[:, 1]]).sum(dim=1).cpu()
            neg_scores_list.append(s)

        neg_scores = torch.stack(neg_scores_list, dim=0)
        return pos_scores, neg_scores
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
