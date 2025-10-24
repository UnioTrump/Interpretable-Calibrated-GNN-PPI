import torch
from torch import Tensor
from torch.nn import Linear, Dropout, ReLU, Module, Sequential
from .base import GNNEncoder
import config

class IntraModalFusion(Module):
    def __init__(self, semantic_dim: int, geometric_dim: int, output_dim: int, dropout: float = config.DROPOUT):
        super().__init__()
        self.total_input_dim = semantic_dim + geometric_dim
        self.fusion_linear = Linear(self.total_input_dim, output_dim)
        self.dropout_layer = Dropout(p=dropout)
        self.relu = ReLU()
        self.out_dim = output_dim

    def forward(self, semantic_embeds: Tensor, geometric_embeds: Tensor) -> Tensor:
        concatenated_embeds = torch.cat([semantic_embeds, geometric_embeds], dim=-1)
        fused_embeds = self.fusion_linear(concatenated_embeds)
        fused_embeds = self.relu(fused_embeds)
        fused_embeds = self.dropout_layer(fused_embeds)
        return fused_embeds

class DualStreamPPI(Module):
    def __init__(self):
        super().__init__()

        self.semantic_stream = GNNEncoder(
            in_channels=1152, 
            hid_dim=config.FEAT_GNN_HID_DIM,
            edge_dim=config.EDGE_DIM,
            heads=config.HEADS,
            dropout=config.DROPOUT
        )
        self.geometric_stream = GNNEncoder(
            in_channels=config.PE_DIM, 
            hid_dim=config.GEO_GNN_HID_DIM,
            edge_dim=config.EDGE_DIM, 
            heads=config.HEADS,
            dropout=config.DROPOUT
        )
        self.intra_fusion = IntraModalFusion(
            semantic_dim=self.semantic_stream.out_dim,
            geometric_dim=self.geometric_stream.out_dim,
            output_dim=config.Dual_FUSE_DIM
        )

        self.fusion_output_dim = self.intra_fusion.out_dim

        self.classifier = Sequential(
            Linear(self.fusion_output_dim, self.fusion_output_dim // 2),
            ReLU(),
            Dropout(p=config.DROPOUT),
            Linear(self.fusion_output_dim // 2, config.OUT_CHANNELS)
        )

    def forward(self, data):
        fused_embeds = self.feat(data)
        prediction = self.classifier(fused_embeds)
        if prediction.shape[-1] == 1:
            return prediction.squeeze(-1)
        return prediction

    def feat(self, data):
        x = data.x
        pe = data.pe
        adj_t = data.adj_t
        fourier = data.fourier

        semantic_embeds = self.semantic_stream(x, adj_t)
        geometric_embeds = self.geometric_stream(pe, fourier)

        fused_embeds = self.intra_fusion(semantic_embeds, geometric_embeds)

        return fused_embeds