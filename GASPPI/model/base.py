import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, LayerNorm, Dropout, ReLU, Sequential
from torch_geometric.nn import TransformerConv, JumpingKnowledge, global_mean_pool
from torch_geometric.utils import dense_to_sparse
from torch_sparse import SparseTensor
from torch import Tensor
from typing import Optional


class GNNEncoder(torch.nn.Module):
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
                layer_dims[i + 1],
                heads=heads,
                concat=False,
                dropout=dropout,
                edge_dim=self.edge_dim,
                beta=True
            )
            self.convs.append(conv)
            self.norms.append(LayerNorm(layer_dims[i + 1]))

        self.jk = JumpingKnowledge(mode='cat')
        self.out_dim = sum(hidden_dims)

    def forward(self, x: Tensor, adj_t: SparseTensor) -> Tensor:
        if self.in_proj:
            x = self.in_proj(x)

        xs = [x]
        for conv, norm in zip(self.convs, self.norms):
            # Convert adjacency format to edge_index format for TransformerConv
            if hasattr(adj_t, 'coo') and callable(getattr(adj_t, 'coo')):
                # Handle SparseTensor
                row, col, edge_attr = adj_t.coo()
                edge_index = torch.stack([row, col], dim=0).long()

                # Pass edge_index and edge_attr to conv
                if self.edge_dim is not None and edge_attr is not None:
                    x = conv(x, edge_index, edge_attr)
                else:
                    x = conv(x, edge_index)

            elif isinstance(adj_t, torch.Tensor) and adj_t.dim() == 2 and adj_t.shape[0] == adj_t.shape[1]:
                # Handle dense adjacency matrix (square matrix)
                edge_index, edge_attr = dense_to_sparse(adj_t)
                edge_index = edge_index.long()

                if self.edge_dim is not None and edge_attr is not None:
                    # 确保 edge_attr 的维度正确
                    if edge_attr.dim() == 1:
                        edge_attr = edge_attr.unsqueeze(-1)  # 从 [E] 转换为 [E, 1]
                    elif edge_attr.dim() == 2 and edge_attr.shape[1] != self.edge_dim:
                        # 如果维度不匹配，进行适当的变换
                        if edge_attr.shape[1] > self.edge_dim:
                            edge_attr = edge_attr[:, :self.edge_dim]  # 截取前 edge_dim 维
                        else:
                            # 如果维度不足，用零填充
                            padding = torch.zeros(edge_attr.shape[0], self.edge_dim - edge_attr.shape[1], device=edge_attr.device)
                            edge_attr = torch.cat([edge_attr, padding], dim=1)
                    
                    x = conv(x, edge_index, edge_attr)
                else:
                    x = conv(x, edge_index)

            elif isinstance(adj_t, torch.Tensor) and adj_t.dim() == 2 and adj_t.shape[0] == 2:
                # Handle edge_index format (2 x num_edges)
                edge_index = adj_t.long()
                x = conv(x, edge_index)

            elif isinstance(adj_t, torch.Tensor):
                # Handle other tensor formats - try to convert to long
                if adj_t.dtype != torch.long:
                    edge_index = adj_t.long()
                else:
                    edge_index = adj_t
                x = conv(x, edge_index)

            else:
                # Fallback: try to convert whatever we have to proper format
                try:
                    if hasattr(adj_t, 'long'):
                        edge_index = adj_t.long()
                    else:
                        edge_index = torch.tensor(adj_t).long()
                    x = conv(x, edge_index)
                except Exception as e:
                    raise ValueError(
                        f"Cannot convert adj_t to proper format for TransformerConv: {type(adj_t)}, shape: {getattr(adj_t, 'shape', 'unknown')}, error: {e}")

            x = F.relu(x)
            x = norm(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

        return self.jk(xs[1:])


class ProteinGNN(torch.nn.Module):
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