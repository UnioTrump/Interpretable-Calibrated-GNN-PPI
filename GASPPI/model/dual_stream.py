import torch
from torch.nn import Linear, Dropout, ReLU, Sequential
from torch import Tensor
from .base import GNNEncoder
import config
from torch.nn import TransformerEncoder, TransformerEncoderLayer, ModuleList

class IntraModalFusion(torch.nn.Module):
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

class FuseBlock(torch.nn.Module):
    def __init__(self, input_dims: list[int], embed_dim: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Linear projections for each modality to map to a common embed_dim
        self.projection_layers = ModuleList([
            Linear(input_dim, embed_dim) for input_dim in input_dims
        ])
        
        # Transformer Encoder for fusion
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dropout, 
            batch_first=True # Input will be (batch_size, seq_len, feature_dim)
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, modal_features_list: list[Tensor]) -> Tensor:

        projected_features = []
        for i, feature in enumerate(modal_features_list):
            if feature is not None:
                projected_features.append(self.projection_layers[i](feature))

        # Stack features to form (N, num_modalities, embed_dim)
        stacked_features = torch.stack(projected_features, dim=1) 
        
        # Apply Transformer Encoder
        fused_output = self.transformer_encoder(stacked_features)
        
        # Aggregate across modalities (e.g., mean)
        # This reduces (N, num_modalities, embed_dim) to (N, embed_dim)
        return torch.mean(fused_output, dim=1)

class DualStreamPPI(torch.nn.Module):
    """Dual-stream model for PPI prediction."""
    def __init__(self, modal_cfg: list, out_channels: int):
        super().__init__()

        # Collect input dimensions for EarlyFusionTransformer
        x_input_dims = [entry.get('in_channels', 0) for entry in modal_cfg]
        pe_input_dims = [entry.get('pe_dim', config.PE_DIM) for entry in modal_cfg]
        # Assuming fourier_dim is same as pe_dim if not specified
        fourier_input_dims = [entry.get('fourier_dim', config.PE_DIM) for entry in modal_cfg]

        # Early Fusion Transformers
        # Use Dual_FUSE_DIM as embed_dim for early fusion outputs
        self.fusion_x = FuseBlock(
            input_dims=x_input_dims,
            embed_dim=config.Dual_FUSE_DIM,
            num_heads=config.HEADS,
            num_layers=1, # One layer for simplicity, can be configured
            dropout=config.DROPOUT
        )
        self.fusion_pe = FuseBlock(
            input_dims=pe_input_dims,
            embed_dim=config.Dual_FUSE_DIM,
            num_heads=config.HEADS,
            num_layers=1, # One layer for simplicity
            dropout=config.DROPOUT
        )
        self.fusion_fourier = FuseBlock(
            input_dims=fourier_input_dims,
            embed_dim=config.Dual_FUSE_DIM,
            num_heads=config.HEADS,
            num_layers=1, # One layer for simplicity
            dropout=config.DROPOUT
        )

        # Single Semantic Stream
        self.semantic_stream = GNNEncoder(
            in_channels=config.Dual_FUSE_DIM, # Input from fused x
            hid_dim=config.FEAT_GNN_HID_DIM,
            edge_dim=config.EDGE_DIM,
            heads=config.HEADS,
            dropout=config.DROPOUT
        )
        # Single Geometric Stream
        self.geometric_stream = GNNEncoder(
            in_channels=config.Dual_FUSE_DIM, # Input from fused pe
            hid_dim=config.GEO_GNN_HID_DIM,
            edge_dim=config.EDGE_DIM,
            heads=config.HEADS,
            dropout=config.DROPOUT
        )
        # Single Intra-modal Fusion
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
        prediction = self.classifier(fused_embeds) # Corrected from self.MLP(fused_embeds)
        if prediction.shape[-1] == 1:
            return prediction.squeeze(-1)
        return prediction

    def feat(self, data):
        # Collect features for each type across all modalities
        all_modal_x_features = []
        all_modal_pe_features = []
        all_modal_fourier_features = []
        adj_t_data = None

        # Iterate through modal_cfg to get modal names and extract features from data object
        for cfg_entry in self.modal_cfg:
            modal_name = cfg_entry['name']

            current_x = getattr(data, f'{modal_name}_x', None)
            current_pe = getattr(data, f'{modal_name}_pe', None)
            current_fourier = getattr(data, f'{modal_name}_fourier', None)
            current_adj_t = getattr(data, f'{modal_name}_adj_t', None)

            if current_x is not None: all_modal_x_features.append(current_x)
            if current_pe is not None: all_modal_pe_features.append(current_pe)
            if current_fourier is not None: all_modal_fourier_features.append(current_fourier)
            
            # Assuming adj_t is the same across all modalities, take from the first one
            if adj_t_data is None and current_adj_t is not None:
                adj_t_data = current_adj_t

        # Apply Early Fusion Transformers
        fused_x_embeds = self.fusion_x(all_modal_x_features)
        fused_pe_embeds = self.fusion_pe(all_modal_pe_features)
        fused_fourier_embeds = self.fusion_fourier(all_modal_fourier_features)

        # Process Semantic Stream
        semantic_embeds = self.semantic_stream(fused_x_embeds, adj_t_data)
        
        # Process Geometric Stream
        geometric_embeds = self.geometric_stream(fused_pe_embeds, fused_fourier_embeds)

        # Intra-modal Fusion
        fused_embeds = self.intra_fusion(semantic_embeds, geometric_embeds)

        return fused_embeds