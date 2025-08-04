import torch
import torch.nn as nn
from torch.nn import Linear, Dropout, ReLU, Sequential, ModuleList
from torch import Tensor
from torch_geometric.nn import global_mean_pool

from .base import UnifiedEncoderBlock

class H_GNNMambaPPI(nn.Module):
    """
    Hierarchical GNN-Mamba model for PPI prediction, built upon a unified,
    configurable encoder block.
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

        # --- 1. Input Projection Layers ---
        self.atom_proj = Linear(atom_in_channels, hidden_dim)
        self.residue_proj = Linear(residue_in_channels, hidden_dim)
        self.pe_proj = Linear(pe_dim, hidden_dim)

        # --- 2. Atom-level Encoder Stack ---
        self.atom_encoder = ModuleList([
            UnifiedEncoderBlock(
                hidden_dim=hidden_dim, use_gnn=True, use_mamba=True,
                mamba_d_state=mamba_d_state, mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand, heads=heads, dropout=dropout,
                edge_dim=atom_edge_dim
            ) for _ in range(num_atom_layers)
        ])
        
        # --- 3. Residue-level Feature Fusion & Encoder Stack ---
        # This projection unifies the concatenated features before the residue encoder
        self.residue_fusion_proj = Linear(hidden_dim * 3, hidden_dim) # pooled_atom + residue + pe

        self.residue_encoder = ModuleList([
            UnifiedEncoderBlock(
                hidden_dim=hidden_dim, use_gnn=True, use_mamba=True,
                mamba_d_state=mamba_d_state, mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand, heads=heads, dropout=dropout,
                edge_dim=residue_edge_dim
            ) for _ in range(num_residue_layers)
        ])

        # --- 4. Final Classifier ---
        self.classifier = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Dropout(p=dropout),
            Linear(hidden_dim // 2, out_channels)
        )

    def forward(self, data) -> Tensor:
        
        # 1. Project initial features
        atom_x = self.atom_proj(data.atom_x)
        residue_x = self.residue_proj(data.residue_x)
        pe_x = self.pe_proj(data.lap_pe)

        # 2. Run through atom encoder stack
        for block in self.atom_encoder:
            atom_x = block(atom_x, data.atom_adj_t, data.atom_edge_attr)

        # 3. Pool atomic features to residue level
        pooled_atom_x = global_mean_pool(atom_x, data.atom_to_residue_map)
        
        # 4. Prepare input for residue encoder
        # Concatenate residue features, pooled atom features, and positional encodings
        residue_fused_x = torch.cat([residue_x, pooled_atom_x, pe_x], dim=-1)
        residue_x = self.residue_fusion_proj(residue_fused_x)

        # 5. Run through residue encoder stack
        for block in self.residue_encoder:
            residue_x = block(residue_x, data.residue_adj_t, data.residue_edge_attr)
            
        # 6. Classify
        prediction = self.classifier(residue_x)

        if prediction.shape[-1] == 1:
            return prediction.squeeze(-1)
        return prediction 