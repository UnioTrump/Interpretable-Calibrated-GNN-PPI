from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Sequential, Linear, Dropout, ReLU
from torch_sparse import SparseTensor
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_mean
from .base import GatedGNNBlock

class PPI(Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 num_layers: int, heads: int = 1,
                 dropout: float = 0.0):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        # Input projection layer
        self.in_proj = Linear(in_channels, hidden_channels)

        # Deep Gated GNN layers
        self.blocks = ModuleList()
        for _ in range(num_layers):
            block = GatedGNNBlock(hidden_channels, hidden_channels, heads=heads, dropout=dropout)
            self.blocks.append(block)

    def reset_parameters(self):
        self.in_proj.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # Create SparseTensor for GATConv, which is more efficient
        adj_t = SparseTensor(
            row=edge_index[0], col=edge_index[1],
            sparse_sizes=(x.size(0), x.size(0))
        ).t()

        # Input projection
        x = self.in_proj(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pass through GatedGNN blocks
        for block in self.blocks:
            x = block(x, adj_t)
        
        return x

class HierarchicalGNN(torch.nn.Module):
    def __init__(self, atom_in_channels,
                 residue_in_channels, hidden_channels, out_channels,
                 atom_num_layers, residue_num_layers, heads=4, dropout=0.3):
        super().__init__()

        # Atom-level GNN
        self.atom_gnn = PPI(
            in_channels=atom_in_channels,
            hidden_channels=hidden_channels,
            num_layers=atom_num_layers, heads=heads, dropout=dropout
        )

        # Residue-level GNN
        # Input dimension is original residue features + features from atom GNN
        residue_in_dim = residue_in_channels + hidden_channels
        self.residue_gnn = PPI(
            in_channels=residue_in_dim,
            hidden_channels=hidden_channels,
            num_layers=residue_num_layers, heads=heads, dropout=dropout
        )

        # Classifier head after fusing global information
        # Input dimension is [local residue features, global protein features]
        classifier_in_dim = hidden_channels + hidden_channels
        self.classifier = Sequential(
            Linear(classifier_in_dim, hidden_channels),
            ReLU(),
            Dropout(p=dropout),
            Linear(hidden_channels, out_channels)
        )

    def forward(self, data):
        # Unpack data from the batch object, using the correct batch attribute names
        atom_x, atom_edge_index, atom_batch = data.atom_x, data.atom_edge_index, data.atom_x_batch
        residue_x, residue_edge_index, residue_batch = data.residue_x, data.residue_edge_index, data.residue_x_batch
        a2r_map = data.a2r_map
        
        # 1. Atom-level GNN
        atom_out = self.atom_gnn(atom_x, atom_edge_index)

        # 2. Atom to Residue Pooling
        # The `a2r_map` needs to be globally indexed for the batch.
        # It's constructed by `scatter_mean` using the `atom_batch` to offset residue indices.
        # NOTE: This assumes `a2r_map` from the loader is for intra-graph mapping.
        # A more robust way might involve pre-calculating offsets, but scatter_mean handles this.
        pooled_atom_feats = scatter_mean(atom_out, a2r_map, dim=0, dim_size=residue_x.size(0))

        # 3. Concatenate original residue features with pooled atomic features
        residue_x_combined = torch.cat([residue_x, pooled_atom_feats], dim=-1)

        # 4. Residue-level GNN
        residue_out = self.residue_gnn(residue_x_combined, residue_edge_index)

        # 5. Global Information Fusion
        global_protein_feats = global_mean_pool(residue_out, residue_batch)
        
        # Expand global features to match the dimension of residue features for concatenation
        global_protein_feats_expanded = global_protein_feats[residue_batch]

        # Concatenate local and global features
        final_residue_feats = torch.cat([residue_out, global_protein_feats_expanded], dim=-1)

        # 6. Classifier Head
        out = self.classifier(final_residue_feats)
        return out