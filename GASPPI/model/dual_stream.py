import torch
from torch import Tensor
from torch.nn import Linear, ModuleList, Dropout, ReLU, Softmax, Module, Sequential
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

class FuseBlock(Module):
    def __init__(self, input_dims: list[int], embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.projection_layers = ModuleList([
            Linear(input_dim, embed_dim) for input_dim in input_dims
        ])
        
        self.gate_weight_layers = ModuleList([
            Linear(embed_dim, 1) for _ in input_dims
        ])
        self.softmax = Softmax(dim=1)
            
    def forward(self, feat_l: list[Tensor]) -> Tensor:
        projected_features = []
        raw_gate_weights = []
        for i, feature in enumerate(feat_l):
            projected_features.append(self.projection_layers[i](feature))

        for i, proj_feat in enumerate(projected_features):
            raw_gate_weights.append(self.gate_weight_layers[i](proj_feat))

        gate_weights = torch.stack(raw_gate_weights, dim=1)
        normalized_weights = self.softmax(gate_weights)

        fused_output = torch.zeros_like(projected_features[0])

        for i, proj_feat in enumerate(projected_features):
            fused_output += normalized_weights[:, i] * proj_feat

        return fused_output

class DualStreamPPI(Module):
    def __init__(self, modal_cfg: list, out_channels: int):
        super().__init__()

        self.modal_cfg = modal_cfg 
        x_input_dims = [entry.get('in_channels', 0) for entry in modal_cfg]
        pe_input_dims = [entry.get('pe_dim', config.PE_DIM) for entry in modal_cfg]
        
        self.fusion_x = FuseBlock(input_dims=x_input_dims, embed_dim=config.Dual_FUSE_DIM,)
        self.fusion_pe = FuseBlock(input_dims=pe_input_dims, embed_dim=config.Dual_FUSE_DIM,)

        self.semantic_stream = GNNEncoder(
            in_channels=config.Dual_FUSE_DIM, 
            hid_dim=config.FEAT_GNN_HID_DIM,
            edge_dim=config.EDGE_DIM,
            heads=config.HEADS,
            dropout=config.DROPOUT
        )
        self.geometric_stream = GNNEncoder(
            in_channels=config.Dual_FUSE_DIM, 
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
            Linear(self.fusion_output_dim // 2, out_channels)
        )

    def forward(self, data):
        fused_embeds = self.feat(data)
        prediction = self.classifier(fused_embeds)
        if prediction.shape[-1] == 1:
            return prediction.squeeze(-1)
        return prediction

    def feat(self, data):
        _x_features = []
        _pe_features = []
        _adj_t = None
        _fourier = None

        for cfg_entry in self.modal_cfg:
            modal_name = cfg_entry['name']

            x = getattr(data, f'{modal_name}_x', None)
            pe = getattr(data, f'{modal_name}_pe', None)
            fourier = getattr(data, f'{modal_name}_fourier', None)
            adj_t = getattr(data, f'{modal_name}_adj_t', None)

            if x is not None: _x_features.append(x)
            if pe is not None: _pe_features.append(pe)
            
            if adj_t is not None:
                _adj_t = adj_t
            if fourier is not None:
                _fourier = fourier

        fused_x_embeds = self.fusion_x(_x_features)
        fused_pe_embeds = self.fusion_pe(_pe_features)
        
        semantic_embeds = self.semantic_stream(fused_x_embeds, _adj_t)
        
        geometric_embeds = self.geometric_stream(fused_pe_embeds, _fourier)

        fused_embeds = self.intra_fusion(semantic_embeds, geometric_embeds)

        return fused_embeds