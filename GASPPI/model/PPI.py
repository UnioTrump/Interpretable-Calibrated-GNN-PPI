import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, LayerNorm, Dropout, ReLU, Sequential
from torch_geometric.nn import TransformerConv, JumpingKnowledge, global_mean_pool
from torch_sparse import SparseTensor
from torch import Tensor
from typing import Optional


class GNNEncoder(torch.nn.Module):
    """
    A powerful and flexible GNN encoder that utilizes TransformerConv layers,
    LayerNorm, Dropout, and JumpingKnowledge to learn rich node representations.
    """
    def __init__(self, in_channels: int, hidden_dims: list, edge_dim: Optional[int] = None,
                 heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.convs = ModuleList()
        self.norms = ModuleList()

        # If the input dimension doesn't match the first hidden dimension,
        # a linear projection layer is used.
        if in_channels != hidden_dims[0]:
            self.in_proj = Linear(in_channels, hidden_dims[0])
        else:
            self.in_proj = None

        # The effective dimensions for the convolutional layers.
        layer_dims = [hidden_dims[0]] + hidden_dims
        for i in range(len(layer_dims) - 1):
            conv = TransformerConv(
                layer_dims[i],
                layer_dims[i+1],
                heads=heads,
                concat=False, # This is the fix. Averages heads instead of concatenating.
                dropout=dropout,  # Dropout on attention weights
                edge_dim=self.edge_dim,
                beta=True  # A key parameter for better performance
            )
            self.convs.append(conv)
            self.norms.append(LayerNorm(layer_dims[i+1]))

        # Jumping Knowledge to aggregate representations from all layers.
        # 'cat' mode concatenates the feature vectors.
        self.jk = JumpingKnowledge(mode='cat')

        # The final output dimension is the sum of all hidden dimensions.
        self.out_dim = sum(hidden_dims)

    def forward(self, x: Tensor, adj_t: SparseTensor) -> Tensor:
        if self.in_proj:
            x = self.in_proj(x)

        # We collect the output of each layer for the JK aggregation.
        xs = [x]
        for conv, norm in zip(self.convs, self.norms):
            # TransformerConv can directly use the SparseTensor `adj_t`.
            # If `adj_t` contains edge weights in its `value` attribute,
            # and `edge_dim` was set during initialization, the layer will use them.
            x = conv(x, adj_t)
            x = F.relu(x)
            x = norm(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

        # Concatenate the outputs of all layers (excluding the initial projection).
        return self.jk(xs[1:])


class ProteinGNN(torch.nn.Module):
    """
    A hierarchical Graph Neural Network for protein structure analysis.
    It processes atom-level graphs, pools features to the residue level,
    and then processes the resulting residue-level graph to make final predictions.
    """
    def __init__(self,
                 atom_in_channels,
                 residue_in_channels,
                 atom_hidden_dims, residue_hidden_dims,
                 out_channels,
                 heads=4, dropout=0.2):
        super().__init__()

        # A GNN encoder for the atom-level graph.
        self.atom_encoder = GNNEncoder(
            in_channels=atom_in_channels,
            hidden_dims=atom_hidden_dims,
            edge_dim=1,  # Edge features are 1-dimensional (scalar weight)
            heads=heads,
            dropout=dropout
        )

        atom_out_dim = self.atom_encoder.out_dim
        residue_gnn_in_channels = residue_in_channels + atom_out_dim

        # A GNN encoder for the residue-level graph.
        self.residue_encoder = GNNEncoder(
            in_channels=residue_gnn_in_channels,
            hidden_dims=residue_hidden_dims,
            edge_dim=1, # Edge features are 1-dimensional (scalar weight)
            heads=heads,
            dropout=dropout
        )

        residue_out_dim = self.residue_encoder.out_dim

        # A final classifier to make predictions for each residue.
        self.classifier = Sequential(
            Linear(residue_out_dim, residue_out_dim // 2),
            ReLU(),
            Dropout(p=dropout),
            Linear(residue_out_dim // 2, out_channels)
        )

    def forward(self,
                atom_x, atom_adj_t,
                residue_x, residue_adj_t,
                atom_to_residue_map):

        # 1. Generate atom embeddings using the atom-level encoder.
        atom_out = self.atom_encoder(atom_x, atom_adj_t)

        # 2. Pool atom features to the residue level.
        pooled_atom_feats = global_mean_pool(atom_out, atom_to_residue_map)

        # 3. Concatenate pooled atom features with original residue features.
        residue_x_combined = torch.cat([residue_x, pooled_atom_feats], dim=-1)

        # 4. Generate final residue embeddings using the residue-level encoder.
        residue_out = self.residue_encoder(residue_x_combined, residue_adj_t)

        # 5. Classify each residue.
        out = self.classifier(residue_out)

        return out