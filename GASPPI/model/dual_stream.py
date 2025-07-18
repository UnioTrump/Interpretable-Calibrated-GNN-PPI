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
                concat=False,
                dropout=dropout,
                edge_dim=self.edge_dim,
                beta=True  # A key parameter for better performance
            )
            self.convs.append(conv)
            self.norms.append(LayerNorm(layer_dims[i+1]))

        # Jumping Knowledge to aggregate representations from all layers.
        self.jk = JumpingKnowledge(mode='cat')

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
    This is the Feature/Topology Stream of our Dual-Stream model.
    It processes atom-level graphs, pools features to the residue level,
    and then processes the resulting residue-level graph.
    Its goal is to learn rich, feature-based representations of residues.
    """
    def __init__(self,
                 atom_in_channels,
                 residue_in_channels,
                 atom_hidden_dims, residue_hidden_dims,
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

        self.out_dim = self.residue_encoder.out_dim


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

        return residue_out


class GeometricStream(torch.nn.Module):
    """
    This is the Geometric/Positional Stream of our Dual-Stream model.
    It takes the difference of Laplacian Positional Encodings (PE) between
    two nodes of an edge and processes it through an MLP.
    Its goal is to learn a representation of the edge's role and
    position within the overall graph structure.
    """
    def __init__(self, pe_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.mlp = Sequential(
            Linear(pe_dim, hidden_dim),
            ReLU(),
            Dropout(p=dropout),
            Linear(hidden_dim, out_dim)
        )

    def forward(self, lap_pe: Tensor, edge_index: Tensor) -> Tensor:
        """
        Args:
            lap_pe (Tensor): The Laplacian Positional Encodings for all nodes
                           in the graph. Shape: [num_nodes, pe_dim]
            edge_index (Tensor): The edge_index of the graph for which to
                                 compute the geometric features.
                                 Shape: [2, num_edges]

        Returns:
            Tensor: A geometric embedding for each edge.
                    Shape: [num_edges, out_dim]
        """
        # For each edge (u, v), compute the geometric feature pe(u) - pe(v)
        node_u = lap_pe[edge_index[0]]
        node_v = lap_pe[edge_index[1]]
        edge_pe_feat = node_u - node_v

        # Process the geometric feature through the MLP
        geometric_embedding = self.mlp(edge_pe_feat)
        return geometric_embedding


class DualStreamPPI(torch.nn.Module):
    """
    The complete Dual-Stream Protein-Protein Interaction prediction model.
    It combines a feature/topology stream (ProteinGNN) and a
    geometric/positional stream (GeometricStream) to make highly informed
    predictions.
    """
    def __init__(self,
                 # Feature Stream Args
                 atom_in_channels, residue_in_channels,
                 atom_hidden_dims, residue_hidden_dims,
                 # Geometric Stream Args
                 pe_dim, geo_hidden_dim, geo_out_dim,
                 # Fusion & General Args
                 fusion_hidden_dim, out_channels,
                 heads=4, dropout=0.2):
        super().__init__()

        # 1. Initialize the Feature/Topology Stream
        self.feature_stream = ProteinGNN(
            atom_in_channels=atom_in_channels,
            residue_in_channels=residue_in_channels,
            atom_hidden_dims=atom_hidden_dims,
            residue_hidden_dims=residue_hidden_dims,
            heads=heads,
            dropout=dropout
        )

        # 2. Initialize the Geometric/Positional Stream
        self.geometric_stream = GeometricStream(
            pe_dim=pe_dim,
            hidden_dim=geo_hidden_dim,
            out_dim=geo_out_dim,
            dropout=dropout
        )

        # 3. Define the Gated Fusion layers
        feature_stream_out_dim = self.feature_stream.out_dim

        # Layers to project both streams to a common dimension
        self.feature_proj = Linear(2 * feature_stream_out_dim, fusion_hidden_dim)
        self.geo_proj = Linear(geo_out_dim, fusion_hidden_dim)

        # Gating layer
        self.gate_linear = Sequential(
            Linear(2 * fusion_hidden_dim, fusion_hidden_dim),
            ReLU(),
            Linear(fusion_hidden_dim, fusion_hidden_dim),
            torch.nn.Sigmoid()
        )

        # 4. Define the final Prediction Head
        self.fusion_head = Sequential(
            Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            ReLU(),
            Dropout(p=dropout),
            Linear(fusion_hidden_dim // 2, out_channels)
        )

    def forward(self, data) -> Tensor:
        # Extract all necessary data from the batch object
        atom_x, atom_adj_t = data.atom_x, data.atom_adj_t
        residue_x, residue_adj_t = data.residue_x, data.residue_adj_t
        atom_to_residue_map = data.atom_to_residue_map
        lap_pe = data.lap_pe
        edge_index = data.edge_index # The edges we want to predict

        # 1. Get residue embeddings from the feature stream
        residue_embeds = self.feature_stream(
            atom_x, atom_adj_t,
            residue_x, residue_adj_t,
            atom_to_residue_map
        )

        # 2. Get geometric embeddings from the geometric stream
        # Note: we use residue_adj_t.to_edge_index() to get the edges
        # of the residue graph that correspond to the PEs.
        geometric_embeds = self.geometric_stream(
            lap_pe, residue_adj_t.to_edge_index()
        )

        # 3. Assemble node features and perform Gated Fusion
        node_u_embeds = residue_embeds[edge_index[0]]
        node_v_embeds = residue_embeds[edge_index[1]]
        feature_embeds = torch.cat([node_u_embeds, node_v_embeds], dim=-1)

        # Project both stream outputs into a common fusion space
        proj_feat = F.relu(self.feature_proj(feature_embeds))
        proj_geo = F.relu(self.geo_proj(geometric_embeds))

        # Calculate the gate dynamically based on both projections
        gate_input = torch.cat([proj_feat, proj_geo], dim=-1)
        gate = self.gate_linear(gate_input)

        # Apply the gate to fuse the representations
        fused_embeds = (gate * proj_feat) + ((1 - gate) * proj_geo)

        # 4. Make the final prediction using the fused representation
        prediction = self.fusion_head(fused_embeds)
        return prediction 