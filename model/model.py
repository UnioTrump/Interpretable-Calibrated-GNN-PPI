import torch
import torch.nn as nn
from torch_geometric.nn import GPSConv, TransformerConv
from torch import Tensor
from torch_sparse import SparseTensor
import config

device = config.DEVICE

class Gated_Fuse(nn.Module):
    def __init__(self, ad, bd, out_d):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(ad + bd, out_d),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(ad + bd, out_d)

    def forward(self, aa, esm):
        x = torch.cat([aa, esm], dim=-1)
        g = self.gate(x)
        return g * self.proj(x)

class PPIBlock(nn.Module):

    def __init__(self, channels, heads, dropout, edge_dim=1):
        super().__init__()

        self.conv1 = GPSConv(
            channels=channels,
            conv=TransformerConv(in_channels=channels, out_channels=channels,
                                 heads=heads, dropout=dropout, edge_dim=2, concat=False),
            heads=heads,
            dropout=dropout,
            act='GELU'
        )
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        identity = x
        x = self.norm(identity +
                      self.dropout(self.conv1(x, edge_index, edge_attr=edge_attr)))
        return x


class PPI(nn.Module):

    def __init__(self, hid_dim: int, heads: int, dropout: float):
        super().__init__()
        aa_d = 566
        esm_d = 1152
        dssp_d = 14
        blosum_d = 20
        self.f_d = esm_d + aa_d
        self.dropout = dropout
        self.hid_dim = hid_dim
        self.aa_norm = nn.LayerNorm(aa_d)
        self.cd_norm = nn.LayerNorm(dssp_d + blosum_d)
        self.node_norm = nn.LayerNorm(self.hid_dim)
        self.edge_norm = nn.LayerNorm(2)

        self.gated = Gated_Fuse(aa_d+dssp_d+blosum_d, esm_d, out_d=hid_dim)
        self.aa_encoder = nn.Sequential(
            nn.Linear(aa_d, aa_d),
            nn.LayerNorm(aa_d),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.convs = nn.ModuleList()
        for i in range(config.NUM_LAYER):
            self.convs.append(PPIBlock(channels=hid_dim, heads=heads, dropout=dropout, edge_dim=2))

        self.classifier = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim // 2, 1),
        )
        self.act = nn.GELU()

        self._init_parameters()

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, ax: Tensor, bx: Tensor, cx: Tensor, dx: Tensor, adj: SparseTensor):
        """
        Parameters:
            ax: AAINDEX features [N, 567]
            bx: ESM-C features [N, 1152]
            cx: DSSP features [N, 14]
            dx: BLOSUM62 features [N, 20]
            adj: Adjacency matrix in COO format, which dim=2
        """
        row, col, edge_attr = adj.coo()
        edge_index = torch.stack([row, col]).long()
        edge_attr = self.edge_norm(edge_attr)

        cdx = torch.cat([cx, dx], dim=1)  # dim: [N, 34]
        cdx = self.cd_norm(cdx)
        wa = self.act(self.aa_encoder(self.aa_norm(ax)))
        w = torch.cat([wa, cdx], dim=1)  # dim: [N, 600]
        x = self.node_norm(self.gated(w, bx))

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)

        return self.classifier(x)