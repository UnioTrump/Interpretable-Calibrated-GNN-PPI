import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, LayerNorm, Dropout, ReLU, Sequential
from torch_geometric.nn import TransformerConv, JumpingKnowledge, global_mean_pool
from torch_sparse import SparseTensor
from torch import Tensor
from typing import Optional
from .base import ProteinGNN


class FeatureStreamOnlyPPI(torch.nn.Module):
    """
    An ablation model that uses ONLY the feature stream (ProteinGNN) to serve
    as a baseline and help diagnose performance issues.
    """
    def __init__(self,
                 atom_in_channels, residue_in_channels,
                 atom_hidden_dims, residue_hidden_dims,
                 out_channels, heads=4, dropout=0.2,
                 **kwargs): # Absorb unused kwargs from config
        super().__init__()

        self.feature_stream = ProteinGNN(
            atom_in_channels=atom_in_channels,
            residue_in_channels=residue_in_channels,
            atom_hidden_dims=atom_hidden_dims,
            residue_hidden_dims=residue_hidden_dims,
            heads=heads,
            dropout=dropout
        )

        self.classifier = Sequential(
            Linear(self.feature_stream.out_dim, self.feature_stream.out_dim // 2),
            ReLU(),
            Dropout(p=dropout),
            Linear(self.feature_stream.out_dim // 2, out_channels)
        )

    def forward(self, data) -> Tensor:
        feature_embeds = self.feature_stream(
            data.atom_x, data.atom_adj_t,
            data.residue_x, data.residue_adj_t,
            data.atom_to_residue_map
        )
        
        prediction = self.classifier(feature_embeds)

        if prediction.shape[-1] == 1:
            return prediction.squeeze(-1)
        return prediction 