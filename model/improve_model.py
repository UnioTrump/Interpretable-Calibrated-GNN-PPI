import torch
import torch.nn as nn
from torch_geometric.nn import GPSConv, TransformerConv
from torch import Tensor
from torch_sparse import SparseTensor
import config
device = config.DEVICE

class Gated_Fuse(nn.Module):
    def __init__(self, aa_d, esm_d, out_d):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(aa_d + esm_d, out_d),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(aa_d + esm_d, out_d)

    def forward(self, aa, esm):
        x = torch.cat([aa, esm], dim=-1)
        g = self.gate(x)
        return g * self.proj(x)

class PPIBlock(nn.Module):

    def __init__(self, channels, heads, dropout, edge_dim=1):
        super().__init__()

        self.conv1 = GPSConv(
            channels=channels,
            conv=TransformerConv(in_channels=channels, out_channels=channels, heads=heads, dropout=dropout, edge_dim=1, concat=False),
            heads=heads,
            dropout=dropout,
            act='GELU'
        )
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        identity = x
        x = self.norm(identity + self.dropout(self.conv1(x, edge_index, edge_attr=edge_attr)))
        return x


class PPI(nn.Module):

    def __init__(self, hid_dim: int, heads: int, dropout: float):
        super().__init__()
        aa_d=567
        esm_d=1152
        # prot_d=1024
        self.f_d = aa_d+esm_d
        self.dropout = dropout
        self.hid_dim = hid_dim
        self.node_norm = nn.LayerNorm(self.hid_dim)
        self.edge_norm = nn.LayerNorm(1)

        # self.reduction = nn.Linear(self.f_d, hid_dim)
        self.gated=Gated_Fuse(aa_d, esm_d, out_d=hid_dim)
        self.aa_encoder = nn.Sequential(
            nn.Linear(aa_d, aa_d),
            nn.LayerNorm(aa_d),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        # self.trf = nn.TransformerEncoderLayer(d_model=aa_d, nhead=3, dropout=dropout)
        self.convs = nn.ModuleList()
        for i in range(config.NUM_LAYER):
            self.convs.append(PPIBlock(channels=hid_dim, heads=heads, dropout=dropout, edge_dim=1))

        self.classifier = nn.Sequential(
            nn.Linear(hid_dim, hid_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim//2, 1),
        )
        self.act = nn.GELU()

    def forward(self, ax: Tensor, bx: Tensor, cx: Tensor, adj: SparseTensor):
        """
        Parameters:
            ax: AAINDEX features [N, 567]
            bx: ESM-C features [N, 1152]
            cx: ProtTrans features [N, 1024]
            adj: Adjacency matrix in COO format
        """

        row, col, edge_attr = adj.coo()
        edge_attr = edge_attr.view(-1, 1)
        edge_index = torch.stack([row, col]).long()
        edge_attr = self.edge_norm(edge_attr)

        w = self.act(self.aa_encoder(ax))  # [N, hid_dim]
        x = self.node_norm(self.gated(w, bx))      #1024+567=1591
        # x = self.act(self.reduction(x))

        for conv in self.convs:
             x = conv(x, edge_index, edge_attr)

        return self.classifier(x)