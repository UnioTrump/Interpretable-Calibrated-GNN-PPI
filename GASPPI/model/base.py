import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, LayerNorm, Dropout, ReLU, Sequential
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_sparse import SparseTensor
from torch import Tensor
from typing import Optional, Tuple

try:
    # 尝试新的导入路径
    from mamba_ssm.models.mixer_seq_simple import Mamba
    print("load Mamba Done!")
except ImportError:
    try:
        # 尝试旧的导入路径
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


class UnifiedEncoderBlock(nn.Module):
    """
    A unified, configurable block for processing sequence and graph data.
    Can be used for atom-level encoding or residue-level interaction blocks.
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
                 edge_dim: Optional[int] = 1):  # 默认边特征维度为1（欧氏距离）
        super().__init__()
        
        self.use_gnn = use_gnn
        self.use_mamba = use_mamba
        self.hidden_dim = hidden_dim

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
        
        # --- GNN Branch ---
        if self.use_gnn and adj_t is not None:
            gnn_out = self.gnn_layer(x, adj_t, edge_attr)
            x_after_gnn = x + self.dropout(gnn_out)
            x_after_gnn = self.gnn_norm(x_after_gnn)
        else:
            x_after_gnn = x

        # --- Mamba Branch ---
        if self.use_mamba:
            # Mamba expects [batch, sequence_length, d_model]
            if x_after_gnn.dim() == 2:
                x_for_mamba = x_after_gnn.unsqueeze(0)
            else:
                x_for_mamba = x_after_gnn
            
            mamba_processed = self.mamba_layer(x_for_mamba)
            
            if x_after_gnn.dim() == 2:
                mamba_processed = mamba_processed.squeeze(0)
            
            x_after_mamba = x_after_gnn + self.dropout(mamba_processed)
            x_after_mamba = self.mamba_norm(x_after_mamba)
        else:
            x_after_mamba = x_after_gnn

        # --- FFN Fusion ---
        if self.use_gnn and self.use_mamba:
            ffn_out = self.ffn(x_after_mamba)
            x_final = x_after_mamba + self.dropout(ffn_out)
            x_final = self.ffn_norm(x_final)
        else:
            x_final = x_after_mamba

        return x_final 