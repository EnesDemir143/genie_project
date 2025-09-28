import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GlobalAttention, global_mean_pool, global_max_pool
from torch_geometric.data import Data


class DropPath(nn.Module):
    """Stochastic depth (per-sample)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class PreNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        return self.norm(x)


class GraphTransformerBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        edge_dim: int | None = None,
        dropout: float = 0.0,
        droppath: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.pre_attn = PreNorm(in_dim)
        self.attn = TransformerConv(
            in_channels=in_dim,
            out_channels=out_dim // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            beta=True,          
            bias=bias,
        )
        self.res_proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)
        self.drop_path1 = DropPath(droppath)
        self.drop_path2 = DropPath(droppath)
        self.ffn = nn.Sequential(
            PreNorm(out_dim),
            nn.Linear(out_dim, out_dim * 4, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 4, out_dim, bias=bias),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        h = self.pre_attn(x)
        h = self.attn(h, edge_index, edge_attr)          # [num_nodes, out_dim]
        x = self.res_proj(x) + self.drop_path1(self.dropout(h))
        h2 = self.ffn(x)
        x = x + self.drop_path2(self.dropout(h2))
        return x

# --- model ---

class GraphTransformerNet(nn.Module):
    """
    Benchmark seviyesinde Graph Transformer.
    Beklenen alanlar:
      data.x: [N, in_dim]
      data.edge_index: [2, E]
      data.edge_attr (ops.): [E, edge_dim]
      data.pe (ops.): [N, pe_dim]  # RWSE/LapPE gibi
      data.batch: [N]
    """
    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        layers: int = 6,
        heads: int = 4,
        edge_dim: int | None = None,
        pe_dim: int = 0,
        dropout: float = 0.1,
        droppath: float = 0.0,
        num_classes: int = 2,     
        task_type: str = "binary", 
        readout: str = "attn",     
        bias: bool = True,
    ):
        super().__init__()
        assert task_type in {"binary", "multiclass"}
        assert readout in {"attn", "meanmax"}

        self.task_type = task_type
        self.num_classes = num_classes
        self.use_edge = edge_dim is not None and edge_dim > 0
        self.pe_dim = int(pe_dim)

        in_total = in_dim + (self.pe_dim if self.pe_dim > 0 else 0)
        self.input_proj = nn.Linear(in_total, hidden, bias=bias)

        # derinlik boyunca lineer artan droppath
        dp_rates = torch.linspace(0, droppath, steps=layers).tolist()
        blocks = []
        for i in range(layers):
            blocks.append(
                GraphTransformerBlock(
                    in_dim=hidden,
                    out_dim=hidden,
                    heads=heads,
                    edge_dim=edge_dim if self.use_edge else None,
                    dropout=dropout,
                    droppath=dp_rates[i],
                    bias=bias,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if readout == "attn":
            self.readout = GlobalAttention(
                gate_nn=nn.Sequential(
                    nn.Linear(hidden, hidden // 2),
                    nn.GELU(),
                    nn.Linear(hidden // 2, 1),
                )
            )
            out_dim_after_pool = hidden
        else:  # mean + max concat
            self.readout = None
            out_dim_after_pool = hidden * 2

        self.head = nn.Sequential(
            nn.LayerNorm(out_dim_after_pool),
            nn.Linear(out_dim_after_pool, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1 if task_type == "binary" else num_classes),
        )

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if x is None:
            raise ValueError("data.x gerekli.")
        if self.pe_dim > 0:
            if getattr(data, "pe", None) is None:
                raise ValueError("Model pe_dim>0 bekliyor ama data.pe yok.")
            x = torch.cat([x, data.pe[:, : self.pe_dim]], dim=-1)

        x = self.input_proj(x)
        edge_attr = data.edge_attr if self.use_edge else None

        for blk in self.blocks:
            x = blk(x, edge_index, edge_attr)

        if self.readout is not None:
            g = self.readout(x, batch)  # [B, hidden]
        else:
            g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)

        logits = self.head(g)
        if self.task_type == "binary":
            return logits.view(-1)  # [B]
        else:
            return logits           # [B, num_classes]
