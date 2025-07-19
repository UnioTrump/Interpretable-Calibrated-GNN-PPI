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

        if in_channels != hidden_dims[0]:
            self.in_proj = Linear(in_channels, hidden_dims[0])
        else:
            self.in_proj = None

        layer_dims = [hidden_dims[0]] + hidden_dims
        for i in range(len(layer_dims) - 1):
            conv = TransformerConv(
                layer_dims[i],
                layer_dims[i+1],
                heads=heads,
                concat=False,
                dropout=dropout,
                edge_dim=self.edge_dim,
                beta=True
            )
            self.convs.append(conv)
            self.norms.append(LayerNorm(layer_dims[i+1]))

        self.jk = JumpingKnowledge(mode='cat')
        self.out_dim = sum(hidden_dims)

    def forward(self, x: Tensor, adj_t: SparseTensor) -> Tensor:
        if self.in_proj:
            x = self.in_proj(x)

        xs = [x]
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = norm(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        
        return self.jk(xs[1:])


class ProteinGNN(torch.nn.Module):
    """
    The Feature/Topology Stream of our Dual-Stream model.
    It processes atom-level graphs, pools features to the residue level,
    and then processes the resulting residue-level graph.
    """
    def __init__(self,
                 atom_in_channels,
                 residue_in_channels,
                 atom_hidden_dims, residue_hidden_dims,
                 heads=4, dropout=0.2):
        super().__init__()

        self.atom_encoder = GNNEncoder(
            in_channels=atom_in_channels,
            hidden_dims=atom_hidden_dims,
            edge_dim=1,
            heads=heads,
            dropout=dropout
        )

        atom_out_dim = self.atom_encoder.out_dim
        residue_gnn_in_channels = residue_in_channels + atom_out_dim

        self.residue_encoder = GNNEncoder(
            in_channels=residue_gnn_in_channels,
            hidden_dims=residue_hidden_dims,
            edge_dim=1,
            heads=heads,
            dropout=dropout
        )

        self.out_dim = self.residue_encoder.out_dim

    def forward(self,
                atom_x, atom_adj_t,
                residue_x, residue_adj_t,
                atom_to_residue_map):

        atom_out = self.atom_encoder(atom_x, atom_adj_t)
        pooled_atom_feats = global_mean_pool(atom_out, atom_to_residue_map)
        residue_x_combined = torch.cat([residue_x, pooled_atom_feats], dim=-1)
        residue_out = self.residue_encoder(residue_x_combined, residue_adj_t)

        return residue_out


class GatedFusion(torch.nn.Module):
    """
    A gated fusion mechanism to adaptively combine feature and geometric streams.
    It learns a gate vector to weigh the importance of each stream's features.
    """
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
    """
    The complete Dual-Stream PPI prediction model. It combines a feature stream
    and a geometric stream to make predictions for each residue (node).
    """
    def __init__(self,
                 atom_in_channels, residue_in_channels,
                 atom_hidden_dims, residue_hidden_dims,
                 pe_dim, geo_hidden_dims,
                 fusion_hidden_dim,
                 out_channels, heads=4, dropout=0.2):
        super().__init__()

        # Feature/Topology Stream
        self.feature_stream = ProteinGNN(
            atom_in_channels=atom_in_channels,
            residue_in_channels=residue_in_channels,
            atom_hidden_dims=atom_hidden_dims,
            residue_hidden_dims=residue_hidden_dims,
            heads=heads,
            dropout=dropout
        )

        # Geometric/Positional Stream (processes PEs on the residue graph)
        self.geometric_stream = GNNEncoder(
            in_channels=pe_dim,
            hidden_dims=geo_hidden_dims,
            edge_dim=None,  # PEs are node features, not edge dependent
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

    def forward(self, data) -> Tensor:
        feature_embeds = self.feature_stream(
            data.atom_x, data.atom_adj_t,
            data.residue_x, data.residue_adj_t,
            data.atom_to_residue_map
        )
        
        # only need the topology, not the edge weights, remove the edge attributes (value).
        geo_adj_t = data.residue_adj_t.clone().set_value_(None, layout='coo')
        geo_embeds = self.geometric_stream(data.lap_pe, geo_adj_t)
        
        fused_embeds = self.fusion(feature_embeds, geo_embeds)
        
        prediction = self.classifier(fused_embeds)

        # Squeeze the last dimension to match target shape [N] for loss calculation
        if prediction.shape[-1] == 1:
            return prediction.squeeze(-1)
        return prediction


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