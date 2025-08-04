import torch
import torch.nn as nn
from torch.nn import Linear, Dropout, ReLU, Sequential, ModuleList
from torch import Tensor
from torch_geometric.nn import global_mean_pool

from .base import InteractionBlock

class H_GNNMambaPPI(nn.Module):
    """
    Hierarchical GNN-Mamba model for PPI prediction, featuring a dual-stream
    architecture at the residue level.
    """
    def __init__(self,
                 atom_in_channels: int,
                 residue_in_channels: int,
                 pe_dim: int,
                 hidden_dim: int,
                 num_atom_layers: int,
                 num_residue_layers: int,
                 out_channels: int,
                 mamba_d_state: int = 16,
                 mamba_d_conv: int = 4,
                 mamba_expand: int = 2,
                 heads: int = 4,
                 dropout: float = 0.2,
                 atom_edge_dim: int = 1,
                 residue_edge_dim: int = 1):
        super().__init__()

        # --- Input Projection Blocks ---
        self.residue_proj = nn.Sequential(
            Linear(residue_in_channels, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            Linear(hidden_dim * 2, hidden_dim)
        )
        self.atom_proj = nn.Sequential(
            Linear(atom_in_channels, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim)
        )
        self.pe_proj = Linear(pe_dim, hidden_dim)

        # --- Atom-level Encoder ---
        self.atom_encoder = ModuleList([
            InteractionBlock(
                hidden_dim=hidden_dim, use_gnn=True, use_mamba=True,
                mamba_d_state=mamba_d_state, mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand, heads=heads, dropout=dropout,
                edge_dim=atom_edge_dim
            ) for _ in range(num_atom_layers)
        ])
        
        # --- Residue-level Dual-Stream Encoders ---
        self.feature_fusion_proj = Linear(hidden_dim * 2, hidden_dim)
        
        self.feature_encoder = ModuleList([
            InteractionBlock(
                hidden_dim=hidden_dim, use_gnn=True, use_mamba=True,
                mamba_d_state=mamba_d_state, mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand, heads=heads, dropout=dropout,
                edge_dim=residue_edge_dim
            ) for _ in range(num_residue_layers)
        ])

        self.geometry_encoder = ModuleList([
            InteractionBlock(
                hidden_dim=hidden_dim, use_gnn=True, use_mamba=False,
                mamba_d_state=mamba_d_state, mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand, heads=heads, dropout=dropout,
                edge_dim=residue_edge_dim
            ) for _ in range(num_residue_layers)
        ])

        # --- Final Classifier ---
        self.classifier = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Dropout(p=dropout),
            Linear(hidden_dim // 2, out_channels)
        )

    def forward(self, data) -> Tensor:
        
        atom_x = self.atom_proj(data.atom_x)
        residue_x = self.residue_proj(data.residue_x)
        pe_x = self.pe_proj(data.lap_pe)

        for block in self.atom_encoder:
            atom_x = block(atom_x, data.atom_adj_t, data.atom_edge_attr)

        pooled_atom_x = global_mean_pool(atom_x, data.atom_to_residue_map)
        
        # Feature Stream
        feature_input = torch.cat([residue_x, pooled_atom_x], dim=-1)
        feature_input = self.feature_fusion_proj(feature_input)
        
        for block in self.feature_encoder:
            feature_input = block(feature_input, data.residue_adj_t, data.residue_edge_attr)

        # Geometry Stream
        for block in self.geometry_encoder:
            pe_x = block(pe_x, data.residue_adj_t, data.residue_edge_attr)
            
        final_embedding = feature_input + pe_x
            
        prediction = self.classifier(final_embedding)

        if prediction.shape[-1] == 1:
            return prediction.squeeze(-1)
        return prediction 