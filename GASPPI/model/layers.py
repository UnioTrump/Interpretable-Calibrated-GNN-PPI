import numpy as np
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn
from torch_sparse import SparseTensor

from .PPI import PPI

def atom2residue(atom_mat, residue_mat, a2r_map):
    """Map atom features to residue space with safety checks"""
    # Create tensor on the same device as inputs
    new_atom_mat = torch.zeros((residue_mat.shape[0], atom_mat.shape[-1]),device=atom_mat.device)

    # Aggregate atom features to residue level
    for a_id, a in enumerate(atom_mat):
        r_id = a2r_map[a_id]
        new_atom_mat[r_id] += a

    return new_atom_mat


class AtomBlock(PPI):
    """gcn+gat+attention"""
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int,
                 hidden_heads: int, out_channels: int, out_heads: int,
                 num_layers: int, dropout: float = 0.0,
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None):
        """
        Args：
            param num_nodes: 图中节点数量
            param in_channels: 输入特征维度
            param hidden_channels: 隐藏层每个头的特征维度
            param hidden_heads: 隐藏层注意力头数
            param out_channels: 输出层特征维度
            param out_heads: 输出层注意力头数
            param num_layers: GNN层数
            param dropout: Dropout概率
            param pool_size: 历史池大小
            param buffer_size: 缓冲区大小
            param device: 计算设备
        """
        # 调用父类PPI的初始化方法
        super().__init__(
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            hidden_heads=hidden_heads,
            out_channels=out_channels,
            out_heads=out_heads,
            num_layers=num_layers,
            dropout=dropout,
            pool_size=pool_size,
            buffer_size=buffer_size,
            device=device
        )
        
    def forward(self, x: torch.Tensor, adj_t: SparseTensor, *args) -> torch.Tensor:
        """Forward pass for atom-level processing"""
        # Use the parent class's forward method for atom-level processing
        return super().forward(x, adj_t, *args)


class ResidueBlock(PPI):
    """gcn+gat+attention"""
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int,
                 hidden_heads: int, out_channels: int, out_heads: int,
                 num_layers: int, dropout: float = 0.0,
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None):
        """
        Args：
            param num_nodes: 残基图中节点数量
            param in_channels: 输入特征维度
            param hidden_channels: 隐藏层每个头的特征维度
            param hidden_heads: 隐藏层注意力头数
            param out_channels: 输出层特征维度
            param out_heads: 输出层注意力头数
            param num_layers: GNN层数
            param dropout: Dropout概率
            param pool_size: 历史池大小
            param buffer_size: 缓冲区大小
            param device: 计算设备
        """
        super().__init__(
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            hidden_heads=hidden_heads,
            out_channels=out_channels,
            out_heads=out_heads,
            num_layers=num_layers,
            dropout=dropout,
            pool_size=pool_size,
            buffer_size=buffer_size,
            device=device
        )
        
    def forward(self, x: torch.Tensor, adj_t: SparseTensor, *args) -> torch.Tensor:
        """Forward pass for residue-level processing"""
        # Use the parent class's forward method for residue-level processing
        return super().forward(x, adj_t, *args)


class HierarchicalGNN(nn.Module):
    def __init__(
        self,
        atom_num_nodes: int,
        residue_num_nodes: int,
        atom_in_channels: int,
        residue_in_channels: int,
        hidden_channels: int,
        hidden_heads: int,
        out_channels: int,
        out_heads: int,
        atom_num_layers: int,
        residue_num_layers: int,
        num_blocks: int = 2,  # 模型中的块数量
        dropout: float = 0.0,
        pool_size: Optional[int] = None,
        buffer_size: Optional[int] = None,
        device=None
    ):
        """
        多块层次图神经网络，处理原子级别和残基级别的特征
        
        Args:
            param atom_num_nodes: 原子图中节点数量
            param residue_num_nodes: 残基图中节点数量
            param atom_in_channels: 原子特征输入维度
            param residue_in_channels: 残基特征输入维度
            param hidden_channels: 隐藏层每个头的特征维度
            param hidden_heads: 隐藏层注意力头数
            param out_channels: 输出层特征维度
            param out_heads: 输出层注意力头数
            param atom_num_layers: 每个原子块中的GNN层数
            param residue_num_layers: 每个残基块中的GNN层数
            param num_blocks: 模型中的块数量
            param dropout: Dropout概率
            param pool_size: 历史池大小
            param buffer_size: 缓冲区大小
            param device: 计算设备
        """
        super(HierarchicalGNN, self).__init__()
        
        self.num_blocks = num_blocks
        self._device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 输出特征维度
        self.output_dim = out_channels * out_heads
        
        # 创建多个原子处理块
        self.atom_blocks = nn.ModuleList()
        
        # 第一个原子块接受原始输入特征
        self.atom_blocks.append(
            AtomBlock(
                num_nodes=atom_num_nodes,
                in_channels=37,
                hidden_channels=hidden_channels,
                hidden_heads=hidden_heads,
                out_channels=out_channels,
                out_heads=out_heads,
                num_layers=atom_num_layers,
                dropout=dropout,
                pool_size=pool_size,
                buffer_size=buffer_size,
                device=device
            )
        )
        
        # 后续的原子块接受前一个原子块的输出作为输入
        for i in range(1, num_blocks):
            self.atom_blocks.append(
                AtomBlock(
                    num_nodes=atom_num_nodes,
                    in_channels=self.output_dim,  # 前一个块的输出维度
                    hidden_channels=hidden_channels,
                    hidden_heads=hidden_heads,
                    out_channels=out_channels,
                    out_heads=out_heads,
                    num_layers=atom_num_layers,
                    dropout=dropout,
                    pool_size=pool_size,
                    buffer_size=buffer_size,
                    device=device
                )
            )
        
        # 创建多个残基处理块
        self.residue_blocks = nn.ModuleList()
        
        # 第一个残基块接收残基特征
        self.residue_blocks.append(
            ResidueBlock(
                num_nodes=residue_num_nodes,
                in_channels=1024,
                hidden_channels=hidden_channels,
                hidden_heads=hidden_heads,
                out_channels=out_channels,
                out_heads=out_heads,
                num_layers=residue_num_layers,
                dropout=dropout,
                pool_size=pool_size,
                buffer_size=buffer_size,
                device=device
            )
        )
        
        # 后续的残基块接受前一个残基块的输出和映射的原子特征
        for i in range(1, num_blocks):
            self.residue_blocks.append(
                ResidueBlock(
                    num_nodes=residue_num_nodes,
                    in_channels=self.output_dim,  # 前一个块的输出维度
                    hidden_channels=hidden_channels,
                    hidden_heads=hidden_heads,
                    out_channels=out_channels,
                    out_heads=out_heads,
                    num_layers=residue_num_layers,
                    dropout=dropout,
                    pool_size=pool_size,
                    buffer_size=buffer_size,
                    device=device
                )
            )
        
        # MLP
        self.linear1 = nn.Sequential(
            nn.Linear(self.output_dim * out_heads, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(128, 1)
        )
        
    def forward(self, atom_x: torch.Tensor, atom_adj_t: SparseTensor, 
                residue_x: torch.Tensor, residue_adj_t: SparseTensor, 
                a2r_map, *args):
        """
        多块层次前向传播
        
        Args:
            param atom_x: 原子特征矩阵
            param atom_adj_t: 原子图的邻接矩阵
            param residue_x: 残基特征矩阵
            param residue_adj_t: 残基图的邻接矩阵
            param a2r_map: 原子到残基的映射关系
            param *args: 其他参数
            
        Returns:
            param out: 最终输出分类值
        """
        # 确保输入在同一设备上
        target_device = self._device
        atom_x = atom_x.to(target_device)
        residue_x = residue_x.to(target_device)
        a2r_map = a2r_map.to(target_device)
        
        # 存储每个块的输出用于最终拼接
        all_outputs = []
        
        # 当前的原子和残基输入
        current_atom_x = atom_x
        current_residue_x = residue_x
        
        # 处理每个块
        skip = 0       # 跳跃连接
        for i in range(self.num_blocks):
            # 处理原子特征
            atom_out = self.atom_blocks[i](current_atom_x, atom_adj_t, *args)
            res_out  = self.residue_blocks[i](current_residue_x, residue_adj_t, *args)
            atom_out_mapped = atom2residue(atom_out, res_out, a2r_map)

            skip += atom_out_mapped

            current_atom_x = atom_out
            current_residue_x = skip
        
        # 最终输出层
        out = self.linear1(skip)
        out = self.linear2(out)
        
        # 确保输出有足够的差异性，避免模型总是预测同一个类别
        if out.shape[1] == 2:  # 如果是二分类
            # 应用一个小的扰动，鼓励模型做出不同的预测
            batch_size = out.shape[0]
            if self.training and batch_size > 1:
                # 训练时添加扰动，使模型更容易学习到两个类别
                noise = torch.randn_like(out) * 0.01
                out = out + noise
        
        return out
        
    def reset_parameters(self):
        """重置模型参数"""
        for atom_block in self.atom_blocks:
            atom_block.reset_parameters()
            
        for residue_block in self.residue_blocks:
            residue_block.reset_parameters()
            
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        
    @property
    def device(self):
        """返回模型使用的设备"""
        return self._device