import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, JumpingKnowledge
from torch_sparse import SparseTensor
from torch import Tensor
from typing import Optional, Tuple

from .base import BaseEncoder

class SpectralSimilarityComputer(nn.Module):
    """计算基于特征向量的谱相似性矩阵"""
    def __init__(self, max_nodes: int = 1000):
        super().__init__()
        self.max_nodes = max_nodes
    
    def forward(self, eigenvectors: Tensor) -> Tensor:
        """
        Args:
            eigenvectors: [num_nodes, num_eigenvectors] 拉普拉斯特征向量
        Returns:
            spectral_similarities: [num_eigenvectors, num_nodes, num_nodes] 每个频率下的相似性矩阵
        """
        num_nodes, num_eigenvectors = eigenvectors.shape
        spectral_similarities = []
        
        for k in range(num_eigenvectors):
            u_k = eigenvectors[:, k]  # 第k个特征向量
            # 计算谱相似性: σ_k[i,j] = u_k[i] * u_k[j]
            sigma_k = torch.outer(u_k, u_k)  # [num_nodes, num_nodes]
            spectral_similarities.append(sigma_k)
            
        return torch.stack(spectral_similarities, dim=0)  # [num_eigenvectors, num_nodes, num_nodes]


class FrequencyDecomposer(nn.Module):
    """频域分解器，实现Graph Fourier Transform的频率分析"""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.freq_threshold_learnable = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, node_features: Tensor, eigenvalues: Tensor, eigenvectors: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            node_features: [num_nodes, d_model] 节点特征
            eigenvalues: [num_eigenvectors] 特征值
            eigenvectors: [num_nodes, num_eigenvectors] 特征向量
        Returns:
            low_freq_features, high_freq_features, low_energy, high_energy
        """
        # 归一化特征值到[0,1]
        normalized_eigenvals = eigenvalues / (eigenvalues.max() + 1e-8)
        threshold = torch.sigmoid(self.freq_threshold_learnable)
        
        # 分离低频和高频模式
        low_freq_mask = normalized_eigenvals < threshold
        high_freq_mask = ~low_freq_mask
        
        # Graph Fourier Transform: X̂ = U^T X
        freq_coeffs = torch.matmul(eigenvectors.T, node_features)  # [num_eigenvectors, d_model]
        
        # 分离低频和高频系数
        low_freq_coeffs = freq_coeffs * low_freq_mask.unsqueeze(-1)
        high_freq_coeffs = freq_coeffs * high_freq_mask.unsqueeze(-1)
        
        # 逆变换回空间域: X = U X̂
        low_freq_features = torch.matmul(eigenvectors, low_freq_coeffs)
        high_freq_features = torch.matmul(eigenvectors, high_freq_coeffs)
        
        # 计算频率能量
        low_energy = torch.sum(low_freq_coeffs ** 2, dim=-1).mean()
        high_energy = torch.sum(high_freq_coeffs ** 2, dim=-1).mean()
        
        return low_freq_features, high_freq_features, low_energy, high_energy


class MultiScaleFrequencyWeightNet(nn.Module):
    """多尺度频率重要性学习网络"""
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # 为每个注意力头学习不同的频率重要性
        self.frequency_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, d_model // 4),
                nn.ReLU(),
                nn.Linear(d_model // 4, d_model // 8),
                nn.ReLU(),
                nn.Linear(d_model // 8, 1),
                nn.Sigmoid()
            ) for _ in range(num_heads)
        ])
        
        # 全局频率重要性调节
        self.global_freq_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, eigenvalues: Tensor, head_idx: int) -> Tensor:
        """
        Args:
            eigenvalues: [num_eigenvectors] 拉普拉斯特征值
            head_idx: 当前注意力头的索引
        Returns:
            frequency_weights: [num_eigenvectors] 频率重要性权重
        """
        # 归一化特征值
        normalized_eigenvals = eigenvalues / (eigenvalues.max() + 1e-8)
        
        # 为每个特征值计算重要性权重
        freq_weights = []
        for eigenval in normalized_eigenvals:
            weight = self.frequency_nets[head_idx](eigenval.unsqueeze(0).unsqueeze(0))
            freq_weights.append(weight.squeeze())
            
        frequency_weights = torch.stack(freq_weights) * self.global_freq_weight
        return frequency_weights


class SpectralConv(nn.Module):
    """集成了多尺度谱感知注意力的图卷积层"""
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4, 
                 dropout: float = 0.2, edge_dim: Optional[int] = None, beta: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        
        # TransformerConv作为基础
        self.base_conv = TransformerConv(
            in_channels, out_channels, heads=heads,
            concat=False, dropout=dropout, edge_dim=edge_dim, beta=beta
        )
        
        # 谱感知组件
        self.spectral_similarity_computer = SpectralSimilarityComputer()
        self.frequency_decomposer = FrequencyDecomposer(out_channels)
        self.frequency_weight_nets = MultiScaleFrequencyWeightNet(out_channels, heads)
        
        # 自适应融合门
        self.adaptive_gate = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.Sigmoid()
        )
        
        # 跨尺度信息融合
        self.cross_scale_attention = nn.MultiheadAttention(
            out_channels, num_heads=heads, dropout=dropout, batch_first=True
        )
    
    def forward(self, x: Tensor, edge_index_or_adj: SparseTensor, 
                eigenvalues: Tensor, eigenvectors: Tensor) -> Tensor:
        """
        Args:
            x: [num_nodes, in_channels] 节点特征
            edge_index_or_adj: 边信息
            eigenvalues: [num_eigenvectors] 拉普拉斯特征值
            eigenvectors: [num_nodes, num_eigenvectors] 拉普拉斯特征向量
        """
        # 基础TransformerConv输出
        base_output = self.base_conv(x, edge_index_or_adj)
            
        # 多尺度谱感知增强
        num_nodes = x.size(0)
        
        # 1. 计算谱相似性矩阵
        spectral_similarities = self.spectral_similarity_computer(eigenvectors)  # [k, n, n]
        
        # 2. 频域分解
        low_freq, high_freq, low_energy, high_energy = self.frequency_decomposer(
            base_output, eigenvalues, eigenvectors
        )
        
        # 3. 多头谱感知注意力
        enhanced_outputs = []
        for head in range(self.heads):
            # 学习频率重要性权重
            freq_weights = self.frequency_weight_nets(eigenvalues, head)  # [num_eigenvectors]
            
            # 计算增强的注意力权重
            head_attention = torch.zeros(num_nodes, num_nodes, device=x.device)
            for k, (freq_weight, similarity) in enumerate(zip(freq_weights, spectral_similarities)):
                head_attention += freq_weight * similarity
                
            # 应用attention到特征
            head_output = torch.matmul(F.softmax(head_attention, dim=-1), base_output)
            enhanced_outputs.append(head_output)
        
        # 4. 多头输出融合
        multi_head_output = torch.stack(enhanced_outputs, dim=1).mean(dim=1)
        
        # 5. 频率自适应融合
        combined_features = torch.cat([low_freq, high_freq], dim=-1)
        gate_weights = self.adaptive_gate(combined_features)
        
        # 6. 最终输出
        spectral_enhanced_output = (
            multi_head_output * gate_weights + 
            base_output * (1 - gate_weights)
        )
        
        return spectral_enhanced_output


class SpectralEncoder(BaseEncoder):
    """基于多尺度谱感知注意力的GNN编码器"""
    def __init__(self, in_channels: int, hidden_dims: list, edge_dim: Optional[int] = None,
                 heads: int = 4, dropout: float = 0.2):
        self.edge_dim = edge_dim
        super().__init__(in_channels, hidden_dims, heads, dropout)

    def _build_layers(self):
        for i in range(len(self.layer_dims) - 1):
            conv = SpectralConv(
                self.layer_dims[i],
                self.layer_dims[i+1],
                heads=self.heads,
                dropout=self.dropout,
                edge_dim=self.edge_dim,
                beta=True,
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(self.layer_dims[i+1]))

    def forward(self, x: Tensor, adj_t: SparseTensor, 
                eigenvalues: Tensor, eigenvectors: Tensor) -> Tensor:
        return super().forward(x, adj_t, eigenvalues=eigenvalues, eigenvectors=eigenvectors) 