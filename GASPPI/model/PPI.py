from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, Dropout, ReLU
from torch_sparse import SparseTensor
from torch_geometric.nn import global_mean_pool

from .base import ScalableGNN, GatedGNNBlock

class PPI(ScalableGNN):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, heads: int = 1,
                 dropout: float = 0.0, pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None):
        super().__init__(num_nodes, hidden_channels, num_layers, pool_size,
                         buffer_size, device)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout

        # 初始投影层，将输入特征映射到隐藏维度
        self.in_proj = Linear(in_channels, hidden_channels)

        # 深度门控GNN层
        self.blocks = ModuleList()
        for _ in range(num_layers):
            block = GatedGNNBlock(hidden_channels, hidden_channels, heads=heads, dropout=dropout)
            self.blocks.append(block)

        self.reg_modules = ModuleList([self.in_proj]) + self.blocks
        self.nonreg_modules = ModuleList()

    def reset_parameters(self):
        super().reset_parameters()
        self.in_proj.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()

    def forward(self, x: Tensor, adj_t: SparseTensor, *args) -> Tensor:
        # 初始投影
        x = self.in_proj(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 历史嵌入处理 (兼容ScalableGNN)
        # 第一层不拉取历史，但需要将结果推送到history[0]
        x = self.push_and_pull(self.histories[0], x, *args)

        # 通过门控GNN块进行深度信息交互
        for i, block in enumerate(self.blocks):
            x = block(x, adj_t)
            # 在每个块之后，与对应的历史嵌入交互
            if i < self.num_layers - 1:
                x = self.push_and_pull(self.histories[i], x, *args)
        
        return x


class HierarchicalGNN(torch.nn.Module):
    def __init__(self, atom_num_nodes, residue_num_nodes, atom_in_channels,
                 residue_in_channels, hidden_channels, out_channels,
                 atom_num_layers, residue_num_layers, heads=4, dropout=0.3,
                 pool_size=None, buffer_size=None, device=None):
        super().__init__()

        # 原子级别GNN
        self.atom_gnn = PPI(
            num_nodes=atom_num_nodes, in_channels=atom_in_channels,
            hidden_channels=hidden_channels, out_channels=hidden_channels,
            num_layers=atom_num_layers, heads=heads, dropout=dropout,
            pool_size=pool_size, buffer_size=buffer_size, device=device
        )

        # 残基级别GNN
        # 输入维度是原始残基特征+原子GNN输出特征
        residue_in_dim = residue_in_channels + hidden_channels
        self.residue_gnn = PPI(
            num_nodes=residue_num_nodes, in_channels=residue_in_dim,
            hidden_channels=hidden_channels, out_channels=hidden_channels,
            num_layers=residue_num_layers, heads=heads, dropout=dropout,
            pool_size=pool_size, buffer_size=buffer_size, device=device
        )

        # 全局信息融合后的分类头
        # 输入维度是[局部残基特征, 全局蛋白质特征]
        classifier_in_dim = hidden_channels + hidden_channels
        self.classifier = Sequential(
            Linear(classifier_in_dim, hidden_channels),
            ReLU(),
            Dropout(p=dropout),
            Linear(hidden_channels, out_channels)
        )

    def forward(self, atom_x, atom_adj_t, residue_x, residue_adj_t, a2r_map):
        # 1. 原子级别GNN
        atom_out = self.atom_gnn(atom_x, atom_adj_t)

        # 2. 原子到残基池化 (a2r_map是batch_idx)
        pooled_atom_feats = global_mean_pool(atom_out, a2r_map)

        # 3. 拼接原始残基特征和池化后的原子特征
        residue_x_combined = torch.cat([residue_x, pooled_atom_feats], dim=-1)

        # 4. 残基级别GNN (深度门控网络)
        residue_out = self.residue_gnn(residue_x, residue_adj_t)

        # 5. 全局信息融合
        # 创建一个指示每个节点属于哪个图的batch向量 (这里假设一个调用是一个图)
        batch = torch.zeros(residue_out.size(0), dtype=torch.long, device=residue_out.device)
        global_protein_feats = global_mean_pool(residue_out, batch)
        
        # 将全局特征广播到每个残基节点
        global_protein_feats_expanded = global_protein_feats.repeat(residue_out.size(0), 1)

        # 拼接局部和全局特征
        final_residue_feats = torch.cat([residue_out, global_protein_feats_expanded], dim=-1)

        # 6. 分类头
        out = self.classifier(final_residue_feats)
        return out
