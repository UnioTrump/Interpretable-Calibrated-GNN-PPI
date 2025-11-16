import torch
import torch.nn as nn
from torch_geometric.nn import GPSConv, TransformerConv
from torch import Tensor
from torch_sparse import SparseTensor
import config
device = config.DEVICE


class ResidueEmbedding(nn.Module):
    """
    Args:
        aa_d: int, dimension of AAindex
        d_model: int, dimension of model
        nhead: int, number of attention heads
        dim_feedforward: int, dimension of feedforward network
        dropout: float, dropout rate
    """

    def __init__(self, aa_d: int, d_model: int, nhead: int, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.aa_proj = nn.Linear(aa_d, d_model)
        self.pos_scale = nn.Parameter(torch.ones(1))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.out_norm = nn.LayerNorm(d_model)

        # Kaiming 初始化
        self._init_weights()

    def _init_weights(self):

        nn.init.kaiming_normal_(self.aa_proj.weight, mode='fan_in', nonlinearity='relu')
        for module in self.encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')

    def _get_sinusoidal_encoding(self, positions: Tensor) -> Tensor:
        positions = positions.unsqueeze(1).float()  # [N,1]
        d_model = self.d_model
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=positions.device).float() *
            -(torch.log(torch.tensor(10000.0, device=positions.device)) / d_model)
        )
        pe = torch.zeros(positions.size(0), d_model, device=positions.device)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe

    def forward(self, aa_feats: Tensor, pos_ids: Tensor) -> Tensor:
        # aa_feats: [N, aa_d]; pos_ids: [N]
        tokens = self.aa_proj(aa_feats)            # [N, d_model]
        pos = self._get_sinusoidal_encoding(pos_ids) * self.pos_scale
        tokens = tokens + pos                      # [N, d_model]
        seq = tokens.unsqueeze(0)
        enc = self.encoder(seq).squeeze(0)         # [N, d_model]
        return self.out_norm(enc)

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
            conv=TransformerConv(in_channels=channels, out_channels=channels, heads=heads, dropout=dropout, edge_dim=edge_dim, concat=False),
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
        aa_d = 567
        esm_d = 1152
        self.d_model = hid_dim
        self.dropout = dropout
        self.hid_dim = hid_dim

        self.node_norm = nn.LayerNorm(self.hid_dim)
        self.edge_norm = nn.LayerNorm(1)
        self.residue_embedding = ResidueEmbedding(
            aa_d=aa_d,
            d_model=self.d_model,
            nhead=heads,
            dim_feedforward=max(2 * self.d_model, 512),
            dropout=dropout,
        )

        self.gated = Gated_Fuse(self.d_model, esm_d, out_d=hid_dim)

        self.convs = nn.ModuleList()
        for i in range(config.NUM_LAYER):
            self.convs.append(PPIBlock(channels=hid_dim, heads=heads, dropout=dropout, edge_dim=1))

        self.classifier = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim // 2, 1),
        )
        self.act = nn.GELU()

    def forward(self, ax: Tensor, bx: Tensor, cx: Tensor, adj: SparseTensor):

        pos_ids = torch.arange(ax.size(0), device=ax.device)
        w = self.residue_embedding(ax, pos_ids)  # [N, d_model]
        fused = self.gated(w, bx)                 # [N, hid_dim]
        x = self.node_norm(fused)

        row, col, edge_attr = adj.coo()
        edge_attr = edge_attr.view(-1, 1)
        edge_index = torch.stack([row, col]).long()
        edge_attr = self.edge_norm(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)

        return self.classifier(x)