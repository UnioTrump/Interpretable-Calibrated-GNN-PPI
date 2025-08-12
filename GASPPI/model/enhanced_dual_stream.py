import torch
from torch.nn import ModuleList, Linear, ReLU, Sequential, Dropout, Parameter, Sigmoid, MultiheadAttention
from torch_geometric.nn import global_mean_pool
from torch import Tensor
from typing import Optional

from .spectral_attention import SpectralEncoder
from ..utils import adaptive_frequency_weighting


class ProteinEncoder(torch.nn.Module):
    """分层的蛋白质GNN编码器，集成多尺度谱感知注意力"""
    def __init__(self,
                 atom_in_channels,
                 residue_in_channels,
                 atom_hidden_dims, residue_hidden_dims,
                 heads=4, dropout=0.2):
        super().__init__()

        # 原子级编码器
        self.atom_encoder = SpectralEncoder(
            in_channels=atom_in_channels,
            hidden_dims=atom_hidden_dims,
            edge_dim=1,
            heads=heads,
            dropout=dropout,
        )

        atom_out_dim = self.atom_encoder.out_dim
        residue_gnn_in_channels = residue_in_channels + atom_out_dim

        # 残基级编码器
        self.residue_encoder = SpectralEncoder(
            in_channels=residue_gnn_in_channels,
            hidden_dims=residue_hidden_dims,
            edge_dim=1,
            heads=heads,
            dropout=dropout,
        )

        self.out_dim = self.residue_encoder.out_dim

    def forward(self,
                atom_x, atom_adj_t, atom_eigenvalues, atom_eigenvectors,
                residue_x, residue_adj_t, residue_eigenvalues, residue_eigenvectors,
                atom_to_residue_map):

        atom_out = self.atom_encoder(atom_x, atom_adj_t, atom_eigenvalues, atom_eigenvectors)
        
        pooled_atom_feats = global_mean_pool(atom_out, atom_to_residue_map)
        residue_x_combined = torch.cat([residue_x, pooled_atom_feats], dim=-1)
        
        residue_out = self.residue_encoder(
            residue_x_combined, residue_adj_t, 
            residue_eigenvalues, residue_eigenvectors
        )

        return residue_out


class SpectralFusion(torch.nn.Module):
    """多尺度谱感知融合模块"""
    def __init__(self, feature_dim: int, geo_dim: int, hidden_dim: int, 
                 num_scales: int = 3, task_type: str = "general"):
        super().__init__()
        self.num_scales = num_scales
        self.task_type = task_type
        
        self.feature_projections = ModuleList([Linear(feature_dim, hidden_dim) for _ in range(num_scales)])
        self.geo_projections = ModuleList([Linear(geo_dim, hidden_dim) for _ in range(num_scales)])
        
        self.scale_gates = ModuleList([
            Sequential(Linear(hidden_dim * 2, hidden_dim), ReLU(), Linear(hidden_dim, 1), Sigmoid())
            for _ in range(num_scales)
        ])
        
        self.cross_scale_attention = MultiheadAttention(hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
        
        self.task_weights = adaptive_frequency_weighting(task_type)
        self.scale_weights = Parameter(
            torch.tensor([self.task_weights['high_freq'], self.task_weights['mid_freq'], self.task_weights['low_freq']])
        )
        
        self.out_dim = hidden_dim

    def forward(self, feature_embeds: Tensor, geo_embeds: Tensor, 
                spectrum_info: Optional[dict] = None) -> Tensor:
        
        scale_outputs = []
        for i in range(self.num_scales):
            proj_feature = self.feature_projections[i](feature_embeds)
            proj_geo = self.geo_projections[i](geo_embeds)
            gate_input = torch.cat([proj_feature, proj_geo], dim=-1)
            gate = self.scale_gates[i](gate_input)
            scale_output = (1 - gate) * proj_feature + gate * proj_geo
            scale_outputs.append(scale_output)
        
        multi_scale_features = torch.stack(scale_outputs, dim=0).transpose(0, 1)
        
        fused_features, _ = self.cross_scale_attention(
            multi_scale_features, multi_scale_features, multi_scale_features
        )
        
        task_weighted_features = fused_features * self.scale_weights.view(1, -1, 1)
        
        return task_weighted_features.sum(dim=1)


class PPIModel(torch.nn.Module):
    """最终的双流PPI预测模型"""
    def __init__(self,
                 atom_in_channels, residue_in_channels,
                 atom_hidden_dims, residue_hidden_dims,
                 pe_dim, geo_hidden_dims,
                 fusion_hidden_dim,
                 out_channels, heads=4, dropout=0.2,
                 task_type="general"):
        super().__init__()
        self.task_type = task_type

        self.feature_stream = ProteinEncoder(
            atom_in_channels=atom_in_channels,
            residue_in_channels=residue_in_channels,
            atom_hidden_dims=atom_hidden_dims,
            residue_hidden_dims=residue_hidden_dims,
            heads=heads,
            dropout=dropout
        )

        self.geometric_stream = SpectralEncoder(
            in_channels=pe_dim,
            hidden_dims=geo_hidden_dims,
            edge_dim=None,
            heads=heads,
            dropout=dropout,
        )

        self.fusion = SpectralFusion(
            feature_dim=self.feature_stream.out_dim,
            geo_dim=self.geometric_stream.out_dim,
            hidden_dim=fusion_hidden_dim,
            num_scales=3,
            task_type=task_type
        )

        self.classifier = Sequential(
            Linear(self.fusion.out_dim, self.fusion.out_dim // 2),
            ReLU(),
            Dropout(p=dropout),
            Linear(self.fusion.out_dim // 2, out_channels)
        )

    def forward(self, data) -> Tensor:
        eigenvalues = data.eigenvalues
        eigenvectors = data.eigenvectors
        spectrum_info = data.spectrum_info
        
        feature_embeds = self.feature_stream(
            data.atom_x, data.atom_adj_t, eigenvalues, eigenvectors,
            data.residue_x, data.residue_adj_t, eigenvalues, eigenvectors,
            data.atom_to_residue_map
        )

        geo_adj_t = data.residue_adj_t.clone().set_value_(None, layout='coo')
        geo_embeds = self.geometric_stream(
            data.lap_pe, geo_adj_t, eigenvalues, eigenvectors
        )
        
        fused_embeds = self.fusion(feature_embeds, geo_embeds, spectrum_info)
        
        prediction = self.classifier(fused_embeds)

        if prediction.shape[-1] == 1:
            return prediction.squeeze(-1)
        return prediction 