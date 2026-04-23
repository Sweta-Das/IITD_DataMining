"""
GNN model definitions for COL761 Assignment 3, Question 2.

Models
------
GraphSAGE   – node classification  (Graph A: 7-class accuracy)
GAT         – node classification  (Graph B: 2-class ROC-AUC)
GCNEncoder  – node embedding       (Graph C: link prediction)
LinkPredictor – MLP decoder on Hadamard product for link prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv


# ─────────────────────────────────────────────────────────────────────────────
# GraphSAGE for node classification (Graph A)
# ─────────────────────────────────────────────────────────────────────────────

class GraphSAGE(nn.Module):
    """Multi-layer GraphSAGE with BatchNorm + Dropout."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                self.bns.append(nn.BatchNorm1d(dims[i + 1]))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def get_embeddings(self, x, edge_index):
        """Return the last hidden layer (before output linear)."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# GAT for node classification (Graph B)
# ─────────────────────────────────────────────────────────────────────────────

class GAT(nn.Module):
    """Multi-layer GAT with BatchNorm + Dropout."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # Ensure hidden_channels is divisible by heads
        head_dim = max(hidden_channels // heads, 1)
        hidden_channels = head_dim * heads

        for i in range(num_layers):
            if i == 0:
                in_c = in_channels
            else:
                in_c = hidden_channels

            if i < num_layers - 1:
                self.convs.append(
                    GATConv(in_c, head_dim, heads=heads,
                            dropout=dropout, concat=True))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            else:
                # Last layer: single head, output dimension
                self.convs.append(
                    GATConv(in_c, out_channels, heads=1,
                            dropout=dropout, concat=False))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def get_embeddings(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# GCN Encoder for link prediction (Graph C)
# ─────────────────────────────────────────────────────────────────────────────

class GCNEncoder(nn.Module):
    """Multi-layer GCN producing node embeddings for link prediction."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        for i in range(num_layers):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                self.bns.append(nn.BatchNorm1d(dims[i + 1]))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# MLP decoder (Hadamard product) for link prediction
# ─────────────────────────────────────────────────────────────────────────────

class LinkPredictor(nn.Module):
    """MLP on element-wise product of source/destination embeddings."""

    def __init__(self, in_channels, hidden_channels=256, num_layers=2,
                 dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.lins = nn.ModuleList()
        self.bns  = nn.ModuleList()

        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [1]
        for i in range(num_layers):
            self.lins.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                self.bns.append(nn.BatchNorm1d(dims[i + 1]))

    def forward(self, z_src, z_dst):
        x = z_src * z_dst          # Hadamard product
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lins[-1](x).squeeze(-1)
