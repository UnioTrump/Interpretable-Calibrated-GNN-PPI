import torch
from torch.nn import Linear, Dropout, ReLU, Sequential
from torch import Tensor
from .base import GNNEncoder
import config

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

class MultiModalConcatFusion(torch.nn.Module):
    def __init__(self, input_dims: list[int], output_dim: int, dropout: float = config.DROPOUT):
        super().__init__()
        self.total_input_dim = sum(input_dims)
        self.fusion_linear = Linear(self.total_input_dim, output_dim)
        self.dropout_layer = Dropout(p=dropout)
        self.relu = ReLU()
        self.out_dim = output_dim

    def forward(self, *embeds: Tensor) -> Tensor:
        concatenated_embeds = torch.cat(embeds, dim=-1)
        fused_embeds = self.fusion_linear(concatenated_embeds)
        fused_embeds = self.relu(fused_embeds)
        fused_embeds = self.dropout_layer(fused_embeds)
        return fused_embeds

class DualStreamPPI(torch.nn.Module):
    """Dual-stream model for PPI prediction."""
    def __init__(self, 
                 in_channels,
                 pe_dim, fused_dim, out_channels,
                 modal2_in_channels: int = None, modal3_in_channels: int = None,
                 modal2_pe_dim: int = None, modal3_pe_dim: int = None
                 ):
        super().__init__()

        # Feature Stream (Modal 1) - Semantic Part
        self.feature_semantic_stream = GNNEncoder(
            in_channels=in_channels,
            hid_dim=config.FEAT_GNN_HID_DIM,
            edge_dim=config.EDGE_DIM,
            heads=config.HEADS,
            dropout=config.DROPOUT
        )
        # Feature Stream (Modal 1) - Geometric Part
        self.feature_geometric_stream = GNNEncoder(
            in_channels=pe_dim,
            hid_dim=config.GEO_GNN_HID_DIM,
            edge_dim=config.EDGE_DIM,
            heads=config.HEADS,
            dropout=config.DROPOUT
        )
        # Feature Stream (Modal 1) - Intra-modal Fusion
        self.feature_intra_fusion = IntraModalFusion(
            semantic_dim=self.feature_semantic_stream.out_dim,
            geometric_dim=self.feature_geometric_stream.out_dim,
            output_dim=config.Dual_FUSE_DIM
        )

        # Modal 2 Stream
        self.modal2_semantic_stream = None
        self.modal2_geometric_stream = None
        self.modal2_intra_fusion = None
        if modal2_in_channels is not None and modal2_pe_dim is not None:
            self.modal2_semantic_stream = GNNEncoder(
                in_channels=modal2_in_channels,
                hid_dim=config.FEAT_GNN_HID_DIM,
                edge_dim=config.EDGE_DIM,
                heads=config.HEADS,
                dropout=config.DROPOUT
            )
            self.modal2_geometric_stream = GNNEncoder(
                in_channels=modal2_pe_dim,
                hid_dim=config.GEO_GNN_HID_DIM,
                edge_dim=config.EDGE_DIM,
                heads=config.HEADS,
                dropout=config.DROPOUT
            )
            self.modal2_intra_fusion = IntraModalFusion(
                semantic_dim=self.modal2_semantic_stream.out_dim,
                geometric_dim=self.modal2_geometric_stream.out_dim,
                output_dim=config.Dual_FUSE_DIM
            )

        # Modal 3 Stream
        self.modal3_semantic_stream = None
        self.modal3_geometric_stream = None
        self.modal3_intra_fusion = None
        if modal3_in_channels is not None and modal3_pe_dim is not None:
            self.modal3_semantic_stream = GNNEncoder(
                in_channels=modal3_in_channels,
                hid_dim=config.FEAT_GNN_HID_DIM,
                edge_dim=config.EDGE_DIM,
                heads=config.HEADS,
                dropout=config.DROPOUT
            )
            self.modal3_geometric_stream = GNNEncoder(
                in_channels=modal3_pe_dim,
                hid_dim=config.GEO_GNN_HID_DIM,
                edge_dim=config.EDGE_DIM,
                heads=config.HEADS,
                dropout=config.DROPOUT
            )
            self.modal3_intra_fusion = IntraModalFusion(
                semantic_dim=self.modal3_semantic_stream.out_dim,
                geometric_dim=self.modal3_geometric_stream.out_dim,
                output_dim=config.Dual_FUSE_DIM
            )

        # Build the input dimensions for the final multimodal fusion layer
        input_dims_for_fusion = [self.feature_intra_fusion.out_dim]
        if self.modal2_intra_fusion:
            input_dims_for_fusion.append(self.modal2_intra_fusion.out_dim)
        if self.modal3_intra_fusion:
            input_dims_for_fusion.append(self.modal3_intra_fusion.out_dim)

        self.fusion = MultiModalConcatFusion(
            input_dims=input_dims_for_fusion,
            output_dim=config.FUSE_DIM
        )

        self.classifier = Sequential(
            Linear(self.fusion.out_dim, self.fusion.out_dim // 2),
            ReLU(),
            Dropout(p=config.DROPOUT),
            Linear(self.fusion.out_dim // 2, out_channels)
        )

    def forward(self, data):
        fused_embeds = self.feat(data)
        prediction = self.MLP(fused_embeds)
        return prediction

    def feat(self, data):
        # Process Feature Stream (Modal 1)
        feature_semantic_embeds = self.feature_semantic_stream(data.seq_x, data.seq_adj_t)
        feature_geometric_embeds = self.feature_geometric_stream(data.r_pe, data.r_fourier)
        feature_fused_embeds = self.feature_intra_fusion(feature_semantic_embeds, feature_geometric_embeds)

        all_fused_modal_embeds = [feature_fused_embeds]

        # Process Modal 2 Stream
        if self.modal2_semantic_stream and hasattr(data, 'modal2_x') and hasattr(data, 'modal2_adj_t')\
            and hasattr(data, 'modal2_r_pe') and hasattr(data, 'modal2_r_fourier'):
            modal2_semantic_embeds = self.modal2_semantic_stream(data.modal2_x, data.modal2_adj_t)
            modal2_geometric_embeds = self.modal2_geometric_stream(data.modal2_r_pe, data.modal2_r_fourier)
            modal2_fused_embeds = self.modal2_intra_fusion(modal2_semantic_embeds, modal2_geometric_embeds)
            all_fused_modal_embeds.append(modal2_fused_embeds)

        # Process Modal 3 Stream
        if self.modal3_semantic_stream and hasattr(data, 'modal3_x') and hasattr(data, 'modal3_adj_t')\
            and hasattr(data, 'modal3_r_pe') and hasattr(data, 'modal3_r_fourier'):
            modal3_semantic_embeds = self.modal3_semantic_stream(data.modal3_x, data.modal3_adj_t)
            modal3_geometric_embeds = self.modal3_geometric_stream(data.modal3_r_pe, data.modal3_r_fourier)
            modal3_fused_embeds = self.modal3_intra_fusion(modal3_semantic_embeds, modal3_geometric_embeds)
            all_fused_modal_embeds.append(modal3_fused_embeds)

        fused_embeds = self.fusion(*all_fused_modal_embeds)
        return fused_embeds

    def MLP(self, embedding):
        prediction = self.classifier(embedding)
        if prediction.shape[-1] == 1:
            return prediction.squeeze(-1)
        return prediction