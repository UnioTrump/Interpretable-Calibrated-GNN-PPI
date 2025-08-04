import torch
import torch.nn as nn
from torch.nn import Linear, Dropout, ReLU, Sequential, ModuleList
from torch import Tensor
from torch_geometric.nn import global_mean_pool
from torch_sparse import SparseTensor
import torch
from typing import Optional

from .base import InteractionBlock
from .sequence_stream import MambaSequenceEncoder

class H_GNNMambaPPI(nn.Module):
    """
    Hierarchical GNN-Mamba model for PPI prediction, featuring a dual-stream
    architecture at the residue level, now enhanced with a dedicated 1D sequence
    encoder stream.
    """
    def __init__(self,
                 atom_in_channels: int,
                 residue_in_channels: int,  # Kept for compatibility, but its projection is removed
                 pe_dim: int,
                 hidden_dim: int,
                 num_atom_layers: int,
                 num_residue_layers: int,
                 num_seq_layers: int,  # New parameter for sequence encoder
                 vocab_size: int,       # New parameter for sequence encoder
                 out_channels: int,
                 heads: int,
                 dropout: float,
                 mamba_d_state: int,
                 mamba_d_conv: int,
                 mamba_expand: int,
                 atom_edge_dim: Optional[int] = None,
                 residue_edge_dim: Optional[int] = None):
        super().__init__()

        # --- 1D Sequence Encoder Stream ---
        self.sequence_encoder = MambaSequenceEncoder(
            vocab_size=vocab_size,
            embedding_dim=hidden_dim, # Using hidden_dim directly for embedding
            hidden_dim=hidden_dim,
            num_layers=num_seq_layers,
            heads=heads,
            dropout=dropout,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand
        )
        
        # --- 3D Structure Input Projection Blocks ---
        self.atom_proj = nn.Sequential(
            Linear(atom_in_channels, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )
        self.pe_proj = Linear(pe_dim, hidden_dim)

        # --- Atom-level Encoder (3D GNN Stream) ---
        self.atom_encoder = ModuleList([
            InteractionBlock(
                hidden_dim,
                heads,
                dropout,
                use_gnn=True,
                use_mamba=False,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand
            ) for _ in range(num_atom_layers)
        ])

        # --- Fusion and Residue-level Dual-Stream Encoders ---
        # This projection now fuses the sequence embedding and the pooled atom features.
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
                hidden_dim=hidden_dim, use_gnn=True, use_mamba=False, # Mamba is not used for PE
                mamba_d_state=mamba_d_state, mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand, heads=heads, dropout=dropout,
                edge_dim=residue_edge_dim
            ) for _ in range(num_residue_layers)
        ])

        # --- Final Classifier ---
        self.classifier = Sequential(
            Linear(hidden_dim, hidden_dim * 4),
            ReLU(),
            Dropout(p=dropout),
            Linear(hidden_dim * 4, hidden_dim * 2),
            ReLU(),
            Dropout(p=dropout),
            Linear(hidden_dim * 2, out_channels)
        )

    def forward(self, data) -> Tensor:
        
        # --- 1. 1D Sequence Stream ---
        # The data object must now contain 'residue_seq_ids'
        # Assuming residue_seq_ids is [num_residues], needs to be [1, num_residues] for batch processing
        if data.residue_seq_ids.dim() == 1:
            data.residue_seq_ids = data.residue_seq_ids.unsqueeze(0)
        
        # sequence_embedding shape: (1, num_residues, hidden_dim)
        sequence_embedding = self.sequence_encoder(data.residue_seq_ids)
        # Squeeze back to (num_residues, hidden_dim) to match graph node features
        sequence_embedding = sequence_embedding.squeeze(0)

        # --- 2. 3D Atom Stream ---
        atom_x = self.atom_proj(data.atom_x)
        pe_x = self.pe_proj(data.lap_pe)

        for block in self.atom_encoder:
            atom_x = block(atom_x, data.atom_adj_t, data.atom_edge_attr)

        pooled_atom_x = global_mean_pool(atom_x, data.atom_to_residue_map)
        
        # --- 3. Fusion of 1D Sequence and 3D Atom Streams ---
        feature_input = torch.cat([sequence_embedding, pooled_atom_x], dim=-1)
        feature_input = self.feature_fusion_proj(feature_input)
        
        # --- 4. 3D Residue Dual-Stream Processing ---
        # Feature Stream
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