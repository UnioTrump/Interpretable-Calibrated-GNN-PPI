from typing import Tuple, Optional

import numpy as np
import torch
import torch_geometric.utils as utils
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from torch_geometric.data import Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE


def add_gaussian_edge_weights(data: Data, sigma: float = 1.0) -> Data:
    """Computes and adds Gaussian kernel edge weights."""
    node_features = data.x
    edge_index = data.edge_index

    row, col = edge_index
    feat_i = node_features[row]
    feat_j = node_features[col]

    d = torch.sum((feat_i - feat_j) ** 2, dim=-1)
    weights = torch.exp(-d / (2 * sigma ** 2))

    data.edge_attr = weights.unsqueeze(1)
    return data


def add_laplacian_pe(data: Data, pe_dim: int) -> Data:
    """Computes and adds Laplacian Positional Encodings."""
    transform = AddLaplacianEigenvectorPE(
        k=pe_dim,
        attr_name='lap_pe',
        is_undirected=True
    )
    data = transform(data)
    return data


def compute_full_laplacian_spectrum(data: Data, max_eigenvectors: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算完整的拉普拉斯谱（特征值和特征向量）
    
    Args:
        data: PyG数据对象
        max_eigenvectors: 最大特征向量数量，None表示计算所有
    
    Returns:
        eigenvalues: [k] 特征值（升序排列）
        eigenvectors: [num_nodes, k] 特征向量
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    
    # 计算拉普拉斯矩阵
    laplacian = utils.get_laplacian(edge_index, num_nodes=num_nodes, normalization='sym')
    lap_index, lap_values = laplacian
    
    # 转换为scipy稀疏矩阵进行特征分解
    lap_scipy = csr_matrix(
        (lap_values.cpu().numpy(), (lap_index[0].cpu().numpy(), lap_index[1].cpu().numpy())),
        shape=(num_nodes, num_nodes)
    )
    
    # 设置要计算的特征向量数量
    if max_eigenvectors is None:
        k = min(num_nodes - 1, num_nodes)  # 拉普拉斯矩阵的rank是n-1
    else:
        k = min(max_eigenvectors, num_nodes - 1)
    
    # 计算最小的k个特征值和对应的特征向量
    try:
        eigenvalues, eigenvectors = eigsh(lap_scipy, k=k, which='SM', tol=1e-6)
    except:
        # 如果特征分解失败，使用更保守的方法
        eigenvalues = np.zeros(k)
        eigenvectors = np.random.randn(num_nodes, k)
        print(f"Warning: Eigendecomposition failed for graph with {num_nodes} nodes")
    
    return torch.FloatTensor(eigenvalues), torch.FloatTensor(eigenvectors)


def analyze_spectrum_properties(eigenvalues: torch.Tensor, eigenvectors: torch.Tensor) -> dict:
    """
    分析谱的性质，为多尺度分析提供指导
    
    Args:
        eigenvalues: [k] 特征值
        eigenvectors: [num_nodes, k] 特征向量
    
    Returns:
        spectrum_info: 包含谱分析结果的字典
    """
    spectrum_info = {}
    
    # 基本统计信息
    spectrum_info['num_eigenvalues'] = len(eigenvalues)
    spectrum_info['spectral_gap'] = float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0.0
    spectrum_info['max_eigenvalue'] = float(eigenvalues.max())
    
    # 频率分段阈值（自适应）
    normalized_eigenvals = eigenvalues / (eigenvalues.max() + 1e-8)
    
    # 基于谱间隙自动确定分段点
    if len(eigenvalues) > 2:
        gaps = eigenvalues[1:] - eigenvalues[:-1]
        gap_percentiles = torch.quantile(gaps, torch.tensor([0.33, 0.67]))
        
        low_freq_threshold = float(torch.searchsorted(gaps, gap_percentiles[0]) / len(gaps))
        high_freq_threshold = float(torch.searchsorted(gaps, gap_percentiles[1]) / len(gaps))
    else:
        low_freq_threshold = 0.33
        high_freq_threshold = 0.67
    
    spectrum_info['low_freq_threshold'] = low_freq_threshold
    spectrum_info['high_freq_threshold'] = high_freq_threshold
    
    # 频率分段索引
    low_freq_indices = normalized_eigenvals < low_freq_threshold
    mid_freq_indices = (normalized_eigenvals >= low_freq_threshold) & (normalized_eigenvals < high_freq_threshold)
    high_freq_indices = normalized_eigenvals >= high_freq_threshold
    
    spectrum_info['low_freq_indices'] = low_freq_indices
    spectrum_info['mid_freq_indices'] = mid_freq_indices  
    spectrum_info['high_freq_indices'] = high_freq_indices
    
    # 每个频段的能量
    spectrum_info['low_freq_energy'] = float(torch.sum(eigenvalues[low_freq_indices]))
    spectrum_info['mid_freq_energy'] = float(torch.sum(eigenvalues[mid_freq_indices]))
    spectrum_info['high_freq_energy'] = float(torch.sum(eigenvalues[high_freq_indices]))
    
    return spectrum_info


def add_enhanced_spectral_features(data: Data, max_eigenvectors: int = 32) -> Data:
    """
    为数据添加增强的谱特征，包括完整的特征值分解和多尺度分析
    
    Args:
        data: PyG数据对象
        max_eigenvectors: 最大特征向量数量
    
    Returns:
        data: 添加了增强谱特征的数据对象
    """
    # 计算完整谱
    eigenvalues, eigenvectors = compute_full_laplacian_spectrum(data, max_eigenvectors)
    
    # 分析谱性质
    spectrum_info = analyze_spectrum_properties(eigenvalues, eigenvectors)
    
    # 添加到数据对象
    data.eigenvalues = eigenvalues
    data.eigenvectors = eigenvectors
    data.spectrum_info = spectrum_info
    
    # 保持与原有接口的兼容性
    if not hasattr(data, 'lap_pe'):
        pe_dim = min(16, len(eigenvalues))  # 默认PE维度
        data.lap_pe = eigenvectors[:, :pe_dim]
    
    return data


def create_multiscale_adjacency(data: Data, spectrum_info: dict) -> dict:
    """
    基于谱分析创建多尺度邻接矩阵
    
    Args:
        data: PyG数据对象
        spectrum_info: 谱分析信息
    
    Returns:
        multiscale_adj: 包含不同尺度邻接矩阵的字典
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    eigenvalues = data.eigenvalues
    eigenvectors = data.eigenvectors
    
    multiscale_adj = {}
    
    # 原始邻接矩阵
    adj_original = utils.to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    multiscale_adj['original'] = adj_original
    
    # 基于谱重构的多尺度邻接矩阵
    for scale_name, freq_indices in [
        ('low_freq', spectrum_info['low_freq_indices']),
        ('mid_freq', spectrum_info['mid_freq_indices']),
        ('high_freq', spectrum_info['high_freq_indices'])
    ]:
        if torch.sum(freq_indices) > 0:
            # 选择对应频率的特征向量和特征值
            selected_eigenvals = eigenvalues[freq_indices]
            selected_eigenvecs = eigenvectors[:, freq_indices]
            
            # 重构该尺度的拉普拉斯矩阵: L = U Λ U^T
            reconstructed_laplacian = torch.matmul(
                torch.matmul(selected_eigenvecs, torch.diag(selected_eigenvals)),
                selected_eigenvecs.T
            )
            
            # 从拉普拉斯矩阵恢复邻接矩阵 (简化版本)
            degree_matrix = torch.diag(torch.diag(reconstructed_laplacian))
            adj_reconstructed = degree_matrix - reconstructed_laplacian
            
            # 确保非负性和对称性
            adj_reconstructed = torch.clamp(adj_reconstructed, min=0)
            adj_reconstructed = (adj_reconstructed + adj_reconstructed.T) / 2
            
            multiscale_adj[scale_name] = adj_reconstructed
    
    return multiscale_adj


def adaptive_frequency_weighting(task_type: str = "general") -> dict:
    """
    根据不同的PPI预测任务返回自适应的频率权重
    
    Args:
        task_type: 任务类型 ("physical_interaction", "functional_association", "pathway_coregulation", "general")
    
    Returns:
        frequency_weights: 频率权重字典
    """
    weight_configs = {
        "physical_interaction": {
            "high_freq": 0.6,    # 重视局部直接相互作用
            "mid_freq": 0.3,     # 中等重视复合物层面
            "low_freq": 0.1      # 较少重视全局通路
        },
        "functional_association": {
            "high_freq": 0.2,    # 较少重视直接结合
            "mid_freq": 0.4,     # 重视功能模块
            "low_freq": 0.4      # 重视通路层面关联
        },
        "pathway_coregulation": {
            "high_freq": 0.1,    # 最少重视局部相互作用  
            "mid_freq": 0.3,     # 中等重视中间层次
            "low_freq": 0.6      # 最重视全局通路结构
        },
        "general": {
            "high_freq": 0.33,   # 平衡考虑各个尺度
            "mid_freq": 0.34,
            "low_freq": 0.33
        }
    }
    
    return weight_configs.get(task_type, weight_configs["general"])
