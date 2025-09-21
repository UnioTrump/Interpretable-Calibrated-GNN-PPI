import torch
from torch.nn import Linear, Dropout, ReLU, Sequential, Module, ModuleList
from torch import Tensor
from .base import GNNEncoder
from typing import List


class GatedFusion(torch.nn.Module):
    """Gated fusion of two streams."""
    def __init__(self, feature_dim: int, geo_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_proj = Linear(feature_dim, hidden_dim)
        self.geo_proj = Linear(geo_dim, hidden_dim)
        self.gate_linear = Linear(hidden_dim * 2, hidden_dim)
        self.out_dim = hidden_dim

    def forward(self, feature_embeds: Tensor, geo_embeds: Tensor) -> Tensor:
        proj_feature = self.feature_proj(feature_embeds)
        proj_geo = self.geo_proj(geo_embeds)
        
        gate_input = torch.cat([proj_feature, proj_geo], dim=-1)
        gate = torch.sigmoid(self.gate_linear(gate_input))
        
        fused_embeds = (1 - gate) * proj_feature + gate * proj_geo
        return fused_embeds


class DualStream(torch.nn.Module):
    """Dual-stream model for PPI prediction."""
    def __init__(self, sequence_in_channels,pe_dim,fusion_hidden_dim,
                 heads=4, dropout=0.2):
        super().__init__()

        self.feature_stream = GNNEncoder(
            in_channels=sequence_in_channels,
            edge_dim=1,
            hid_dim=256,
            heads=heads,
            dropout=dropout
        )
        self.geometric_stream = GNNEncoder(
            in_channels=pe_dim,
            edge_dim=1,
            hid_dim=64,
            heads=heads,
            dropout=dropout
        )
        self.fusion = GatedFusion(
            feature_dim=self.feature_stream.out_dim,
            geo_dim=self.geometric_stream.out_dim,
            hidden_dim=fusion_hidden_dim
        )
    def forward(self, data):
        fuse = self.feat(data)

        return fuse

    def feat(self, data):
        feature_embeds = self.feature_stream(data.seq_x, data.seq_adj_t)
        geo_embeds = self.geometric_stream(data.r_pe, data.r_fourier)
        fused_embeds = self.fusion(feature_embeds, geo_embeds)
        return fused_embeds

class MultiView(Module):

    def __init__(self,in_channels: int, pe_dim: int, fuse_dim: int,out_channels: int,
                 n_graphs: int, fusion: str = 'concat', heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.n_graphs = n_graphs
        self.blocks = ModuleList([DualStream(in_channels, pe_dim, fuse_dim, heads, dropout) for _ in range(n_graphs)])
        self.fusion = fusion
        if fusion == 'concat': self.out_dim = fuse_dim * n_graphs
        elif fusion == 'mean': self.out_dim = fuse_dim
        else: raise ValueError("fusion must be concat or mean")
        self.classifier = Sequential(
            Linear(self.out_dim, self.out_dim // 2),
            ReLU(),
            Dropout(p=dropout),
            Linear(self.out_dim // 2, out_channels)
        )

    def forward(self, graphs: List):

        assert len(graphs) == self.n_graphs
        embeds = [block(g) for block, g in zip(self.blocks, graphs)]
        if self.fusion == 'concat':
            fused = torch.cat(embeds, dim=-1)
        elif self.fusion == 'mean':
            fused = sum(embeds) / len(embeds)
        else:
            raise ValueError
        return self.MLP(fused)

    def MLP(self, embedding: Tensor) -> Tensor:
        pred = self.classifier(embedding)
        return pred.squeeze(-1) if pred.shape[-1] == 1 else pred