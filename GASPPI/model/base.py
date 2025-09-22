import torch
import torch.nn.functional as F
from torch.nn import ModuleList, LayerNorm, Linear
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import JumpingKnowledge, AntiSymmetricConv, GeneralConv
from torch import Tensor
from typing import Optional

from torch_sparse import SparseTensor

import config

class GNNEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, hid_dim: int, edge_dim: Optional[int] = None,
                 heads: int = config.HEADS, dropout: float = config.DROPOUT):
        super().__init__()
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.hid_dim = hid_dim

        if self.hid_dim != in_channels:
            self.in_proj = Linear(in_channels, self.hid_dim)
        else:
            self.in_proj = None
            
        self.convs = ModuleList()
        self.norms = ModuleList()
        phi = GeneralConv(in_channels=self.hid_dim, out_channels=self.hid_dim, in_edge_channels=config.EDGE_DIM,
                        heads=heads, aggr=config.CONV_AGGR, directed_msg=config.CONV_DIRECTED, l2_normalize=config.CONV_L2_NORM)
        conv1 = AntiSymmetricConv(self.hid_dim, phi=phi)
        self.convs.append(conv1)
        for _ in range(config.NUM_LAYER):
            self.convs.append(conv1)
            self.norms.append(LayerNorm(self.hid_dim))

        self.jk = JumpingKnowledge(mode=config.JK_MODE)
        self.out_dim : int = hid_dim * (config.NUM_LAYER + 1)

    def forward(self, x: Tensor, adj_t: SparseTensor) -> Tensor:
        if self.in_proj is not None:
            x = self.in_proj(x)

        xs = [x]
        for conv, norm in zip(self.convs, self.norms):
            # Convert adjacency format to edge_index format for TransformerConv
            if hasattr(adj_t, 'coo') and callable(getattr(adj_t, 'coo')):
                # Handle SparseTensor
                row, col, edge_attr = adj_t.coo()
                edge_index = torch.stack([row, col], dim=0).long()
                x = conv(x, edge_index, edge_attr)

            elif isinstance(adj_t, torch.Tensor) and adj_t.dim() == 2 and adj_t.shape[1] == adj_t.shape[0]:
                # Handle dense adjacency matrix (square matrix)
                edge_index, edge_attr = dense_to_sparse(adj_t)
                edge_index = edge_index.long()
                if edge_attr.dim() == 1:
                    edge_attr = edge_attr.unsqueeze(-1)
                x = conv(x, edge_index, edge_attr)

            elif isinstance(adj_t, torch.Tensor) and adj_t.dim() == 2 and adj_t.shape[0] == 2:
                edge_index = adj_t.long()
                x = conv(x, edge_index)

            elif isinstance(adj_t, torch.Tensor):
                if adj_t.dtype != torch.long:
                    edge_index = adj_t.long()
                else:
                    edge_index = adj_t
                x = conv(x, edge_index)

            else:
                print("FUCKING data format!!!")
            x = F.relu(x)
            x = norm(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

        out = self.jk(xs)
        return out