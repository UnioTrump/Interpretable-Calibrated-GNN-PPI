import torch
from torch.nn import Linear, Dropout, ReLU, Sequential
from torch import Tensor
from .base import GNNEncoder, ProteinGNN


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


class DualStreamPPI(torch.nn.Module):
    """Dual-stream model for PPI prediction."""
    def __init__(self,
                 atom_in_channels, residue_in_channels,
                 pe_dim,fusion_hidden_dim, out_channels, heads=4, dropout=0.2):
        super().__init__()

        self.feature_stream = ProteinGNN(
            atom_in_channels=atom_in_channels,
            residue_in_channels=residue_in_channels,
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

        self.classifier = Sequential(
            Linear(self.fusion.out_dim, self.fusion.out_dim // 2),
            ReLU(),
            Dropout(p=dropout),
            Linear(self.fusion.out_dim // 2, out_channels)
        )

    def forward(self, data):
        feature_embeds = self.feature_stream(
            data.atom_x, data.atom_adj_t,
            data.residue_x, data.residue_adj_t,
            data.a2r_map
        )

        geo_embeds = self.geometric_stream(data.r_pe, data.r_fourier)
        
        fused_embeds = self.fusion(feature_embeds, geo_embeds)
        
        prediction = self.classifier(fused_embeds)

        if prediction.shape[-1] == 1:
            return prediction.squeeze(-1)
        return prediction

    def feat(self, data):
        feature_embeds = self.feature_stream(
            data.atom_x, data.atom_adj_t,
            data.residue_x, data.residue_adj_t,
            data.a2r_map
        )

        r_fourier = data.r_fourier
        geo_adj_t = r_fourier
        geo_embeds = self.geometric_stream(data.r_pe, geo_adj_t)

        fused_embeds = self.fusion(feature_embeds, geo_embeds)
        return fused_embeds

    def MLP(self, embedding):
        prediction = self.classifier(embedding)

        if prediction.shape[-1] == 1:
            return prediction.squeeze(-1)
        return prediction