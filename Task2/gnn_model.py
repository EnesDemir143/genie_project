import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, GINConv, GINEConv,
    global_mean_pool, global_max_pool, global_add_pool, GlobalAttention
)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)

def make_conv(conv_type, in_dim, out_dim, edge_dim=None, heads=4):
    if conv_type == "gcn":
        return GCNConv(in_dim, out_dim)
    if conv_type == "sage":
        return SAGEConv(in_dim, out_dim)
    if conv_type == "gat":
        return GATConv(in_dim, out_dim, heads=heads, concat=False)
    if conv_type == "gin":
        return GINConv(MLP(in_dim, out_dim, out_dim))
    if conv_type == "gine":
        return GINEConv(MLP(in_dim, out_dim, out_dim), edge_dim=edge_dim)
    raise ValueError(f"Unknown conv_type: {conv_type}")

class GenericGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        num_classes: int = 2,
        conv_type: str = "sage",          # "gcn" | "sage" | "gat" | "gin" | "gine"
        edge_dim: int | None = None,      # just for GINE 
        heads: int = 4,                   # just for GAT 
        norm_type: str = "batch",         # "batch" | "layer" | "none"
        pooling: str = "mean",            # "mean" | "add" | "max" | "attn"
        dropout: float = 0.2,
        residual: bool = True,
    ):
        super().__init__()
        self.conv_type = conv_type
        self.dropout = dropout
        self.residual = residual
        self.use_edge = (conv_type == "gine")

        self.convs = nn.ModuleList()
        self.convs.append(make_conv(conv_type, in_channels, hidden_channels, edge_dim=edge_dim, heads=heads))
        for _ in range(num_layers - 1):
            self.convs.append(make_conv(conv_type, hidden_channels, hidden_channels, edge_dim=edge_dim, heads=heads))

        def make_norm(dim):
            if norm_type == "batch": return nn.BatchNorm1d(dim)
            if norm_type == "layer": return nn.LayerNorm(dim)
            return nn.Identity()
        self.norms = nn.ModuleList([make_norm(hidden_channels) for _ in range(num_layers)])

        if pooling == "mean":
            self.pool = lambda x, batch: global_mean_pool(x, batch)
        elif pooling == "add":
            self.pool = lambda x, batch: global_add_pool(x, batch)
        elif pooling == "max":
            self.pool = lambda x, batch: global_max_pool(x, batch)
        elif pooling == "attn":
            gate_nn = nn.Sequential(nn.Linear(hidden_channels, 1))
            self.attn = GlobalAttention(gate_nn)
            self.pool = lambda x, batch: self.attn(x, batch)
        else:
            raise ValueError("pooling must be one of: mean/add/max/attn")

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        h = x
        for i, conv in enumerate(self.convs):
            h_in = h
            if self.use_edge:
                h = conv(h, edge_index, edge_attr=edge_attr)
            else:
                h = conv(h, edge_index)
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.residual and h.shape == h_in.shape:
                h = h + h_in

        g = self.pool(h, batch)
        out = self.head(g)
        return out
