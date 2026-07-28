import torch
import torch.nn as nn
from torch_geometric.nn import GPSConv, TransformerConv
from torch import Tensor
from torch_sparse import SparseTensor
import config
from timm.layers import DropPath

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

    def __init__(self, channels, heads, dropout, edge_dim):
        super().__init__()

        self.conv1 = GPSConv(
            channels=channels,
            conv=TransformerConv(in_channels=channels, out_channels=channels,
                                 heads=heads, dropout=dropout, edge_dim=edge_dim, concat=False),
            heads=heads,
            dropout=dropout,
            act='GELU'
        )
        self.norm = nn.LayerNorm(channels)
        self.drop_path = DropPath(0.2)

    def forward(self, x, edge_index, edge_attr):
        identity = x
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.drop_path(x)
        x = self.norm(x)
        return x + identity


class PPI(nn.Module):

    def __init__(self, hid_dim: int, heads: int, dropout: float):
        super().__init__()
        aa_d = 566
        esm_d = 1152
        dssp_d = 14
        blosum_d = 20
        pse_d = 1
        res_atom_d = 7
        self.f_d = esm_d + aa_d
        self.dropout = dropout
        self.hid_dim = hid_dim
        self.aa_norm = nn.LayerNorm(aa_d)
        self.cd_norm = nn.LayerNorm(dssp_d + blosum_d + pse_d + res_atom_d)
        self.node_norm = nn.LayerNorm(self.hid_dim)
        self.edge_norm = nn.LayerNorm(1)

        self.gated = Gated_Fuse(aa_d+dssp_d+blosum_d+pse_d+res_atom_d, esm_d, out_d=hid_dim)
        self.aa_encoder = nn.Sequential(
            nn.Linear(aa_d, aa_d),
            nn.LayerNorm(aa_d),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.convs = nn.ModuleList()
        for i in range(config.NUM_LAYER):
            self.convs.append(PPIBlock(channels=hid_dim, heads=heads, dropout=dropout, edge_dim=1))

        self.act = nn.GELU()

        self.classifier = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim // 2, 1),
        )

        self._init_parameters()

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def Fuse(self, ax: Tensor, bx: Tensor, cx: Tensor, dx: Tensor, ex: Tensor, fx: Tensor, adj: SparseTensor):
        row, col, edge_attr = adj.coo()
        edge_index = torch.stack([row, col]).long()
        if len(edge_attr.shape) == 2:
            edge_attr = self.edge_norm(edge_attr[:, 0].unsqueeze(1))
        else:
            edge_attr = self.edge_norm(edge_attr.unsqueeze(-1))

        cdx = torch.cat([cx, dx, ex, fx], dim=1)
        cdx = self.cd_norm(cdx)
        wa = self.act(self.aa_encoder(self.aa_norm(ax)))
        w = torch.cat([wa, cdx], dim=1)
        x = self.node_norm(self.gated(w, bx))
        return x, edge_index, edge_attr

    def Explain(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        """Second-half forward for explainability: conv layers + classifier only.
        Args:
            x: fused node features from Fuse() [N, hid_dim]
            edge_index: [2, E]
            edge_attr: [E, 1]
        """
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        return self.classifier(x)

    def forward(self, ax: Tensor, bx: Tensor, cx: Tensor, dx: Tensor, ex: Tensor, fx: Tensor, adj: SparseTensor):
        """
        Parameters:
            ax: AAINDEX features [N, 566]
            bx: ESM-C features [N, 1152]
            cx: DSSP features [N, 14]
            dx: BLOSUM62 features [N, 20]
            ex: Pseudo position [N, 1]
            fx: Residue atomic features [N, 7]
            adj: Adjacency matrix in COO format, which dim=1
        """
        x, edge_index, edge_attr = self.Fuse(ax, bx, cx, dx, ex, fx, adj)         # Full
        return self.Explain(x, edge_index, edge_attr)

