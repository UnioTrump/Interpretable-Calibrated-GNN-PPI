import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, LayerNorm, Dropout, ReLU, Sequential
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_sparse import SparseTensor
from torch import Tensor
from typing import Optional

try:
    from mamba_ssm.models.mixer_seq_simple import Mamba
    print("load Mamba Done!")
except ImportError:
    try:
        from mamba_ssm import Mamba
        print("load Mamba Done!")
    except ImportError:
        print("load Mamba None!")
        Mamba = None

class TransformerConvLayer(torch.nn.Module):
    """A wrapper for TransformerConv with activation, norm, and dropout."""
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4, dropout: float = 0.2, edge_dim: Optional[int] = None):
        super().__init__()
        self.conv = TransformerConv(
            in_channels,
            out_channels,
            heads=heads,
            concat=False,
            dropout=dropout,
            edge_dim=edge_dim,
            beta=True
        )
        self.norm = LayerNorm(out_channels)
        self.dropout_p = dropout

    def forward(self, x: Tensor, adj_t: SparseTensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        x = self.conv(x, adj_t, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.norm(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x


class InteractionBlock(nn.Module):
    """
    A configurable block for processing sequence and graph data, acting as
    the main building block for the protein encoders.
    """
    def __init__(self,
                 hidden_dim: int,
                 use_gnn: bool = True,
                 use_mamba: bool = True,
                 mamba_d_state: int = 16,
                 mamba_d_conv: int = 4,
                 mamba_expand: int = 2,
                 heads: int = 4,
                 dropout: float = 0.2,
                 edge_dim: Optional[int] = 1):
        super().__init__()
        
        self.use_gnn = use_gnn
        self.use_mamba = use_mamba

        if self.use_gnn:
            self.gnn_layer = TransformerConvLayer(hidden_dim, hidden_dim, heads, dropout, edge_dim=edge_dim)
            self.gnn_norm = LayerNorm(hidden_dim)

        if self.use_mamba:
            if Mamba is None:
                raise ImportError("Mamba-ssm is not installed. Please install it with `pip install mamba-ssm causal-conv1d`")
            self.mamba_layer = Mamba(d_model=hidden_dim, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
            self.mamba_norm = LayerNorm(hidden_dim)

        if self.use_gnn and self.use_mamba:
            self.ffn = Sequential(
                Linear(hidden_dim, hidden_dim * 4),
                ReLU(),
                Linear(hidden_dim * 4, hidden_dim)
            )
            self.ffn_norm = LayerNorm(hidden_dim)
        
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor, adj_t: Optional[SparseTensor] = None, edge_attr: Optional[Tensor] = None) -> Tensor:
        
        # --- GNN Branch with Residual Connection ---
        if self.use_gnn and adj_t is not None:
            shortcut = x
            gnn_out = self.gnn_layer(x, adj_t, edge_attr)
            x = shortcut + self.dropout(gnn_out)
            x = self.gnn_norm(x)

        # --- Mamba Branch with Residual Connection ---
        if self.use_mamba:
            shortcut = x
            
            # Dimension adaptation for Mamba
            is_2d = x.dim() == 2
            if is_2d:
                x = x.unsqueeze(0)
            
            mamba_out = self.mamba_layer(x)
            
            if is_2d:
                mamba_out = mamba_out.squeeze(0)
            
            x = shortcut + self.dropout(mamba_out)
            x = self.mamba_norm(x)

        # --- FFN Block with Residual Connection ---
        if self.use_gnn and self.use_mamba:
            shortcut = x
            ffn_out = self.ffn(x)
            x = shortcut + self.dropout(ffn_out)
            x = self.ffn_norm(x)

        return x 