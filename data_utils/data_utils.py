import pickle
import torch
import os
import numpy as np
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from GASPPI import add_gaussian_edge_weights
import torch_geometric.transforms as T
import config


def frequency_filtering(eigenvalues, x_low, x_high):
    """
    频域滤波函数，生成注意力优化矩阵

    Args:
        eigenvalues: 拉普拉斯矩阵的特征值
        x_low: 低频信号
        x_high: 高频信号

    Returns:
        attention_optimization_matrix: 注意力优化矩阵
    """
    num_nodes = x_low.shape[0]
    # 向量化计算 sum_matrix
    eigenvalues_reshaped_i = eigenvalues.view(-1, 1)
    eigenvalues_reshaped_j = eigenvalues.view(1, -1)
    sum_matrix = eigenvalues_reshaped_i + eigenvalues_reshaped_j

    # 计算每个节点的低频和高频能量
    low_energy = torch.sum(x_low ** 2, dim=1)
    high_energy = torch.sum(x_high ** 2, dim=1)

    # 向量化计算 filter_matrix
    low_energy_reshaped_i = low_energy.view(-1, 1)
    high_energy_reshaped_j = high_energy.view(1, -1)
    denominator = low_energy.sum() + high_energy.sum()

    # 避免除零错误
    if denominator == 0:
        denominator = 1e-8

    filter_matrix = (low_energy_reshaped_i + high_energy_reshaped_j) / denominator

    # 对 sum_matrix 进行滤波
    attention_optimization_matrix = sum_matrix * filter_matrix

    # 将 NaN 替换为 0
    attention_optimization_matrix = torch.nan_to_num(attention_optimization_matrix, nan=0.0)
    return attention_optimization_matrix


def compute_fourier_features(x, edge_index, threshold=1.0):
    """
    计算图的傅里叶特征

    Args:
        x: 节点特征 [num_nodes, num_features]
        edge_index: 边索引 [2, num_edges]
        threshold: 低频/高频分离阈值

    Returns:
        dict: 包含傅里叶特征的字典
    """
    num_nodes = x.shape[0]
    device = x.device

    # 构建邻接矩阵
    edge_adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float, device=device)
    edge_adj[edge_index[0, :], edge_index[1, :]] = 1

    # 图傅里叶变换
    adj = edge_adj + edge_adj.t()  # 确保对称
    adj = adj.fill_diagonal_(0)  # 去除自环

    # 计算度矩阵
    degree = torch.diag(adj.sum(dim=1))
    if torch.any(torch.diag(degree) == 0):
        # 处理孤立节点
        epsilon = 1e-5
        eye_matrix = epsilon * torch.eye(degree.shape[0], device=device)
        degree_inv_sqrt = torch.inverse(torch.sqrt(degree + eye_matrix))
    else:
        degree_inv_sqrt = torch.inverse(torch.sqrt(degree))

    # 计算归一化的拉普拉斯矩阵
    laplacian = torch.eye(num_nodes, device=device) - degree_inv_sqrt @ adj @ degree_inv_sqrt

    # 进行特征分解
    eigenvalues, eigenvectors = torch.linalg.eig(laplacian)
    # 取实部
    eigenvalues = eigenvalues.real.float()
    eigenvectors = eigenvectors.real.float()

    # Graph傅里叶变换
    x_fourier = eigenvectors.t() @ x.float()

    # 分离低频和高频信号
    low_mask = (eigenvalues < threshold).float().unsqueeze(1)
    high_mask = (eigenvalues >= threshold).float().unsqueeze(1)

    x_low = eigenvectors @ (x_fourier * low_mask)
    x_high = eigenvectors @ (x_fourier * high_mask)

    # 生成注意力优化矩阵
    attention_optimization_matrix = frequency_filtering(eigenvalues, x_low, x_high)

    return attention_optimization_matrix


class DataLoader:

    def __init__(self, device=None, enable_fourier=True, fourier_threshold=1.0, enable_random_walk_pe=True,
                 walk_length=16):
        self.device = device or config.DEVICE
        self.enable_fourier = enable_fourier
        self.fourier_threshold = fourier_threshold
        self.enable_random_walk_pe = enable_random_walk_pe
        self.walk_length = walk_length
        # Initialize label mapping for string to integer conversion
        self.label_to_idx = {}
        self.idx_to_label = {}

    @staticmethod
    def load_data(pkl_path):
        return sum((pickle.load(open(os.path.join(pkl_path, f), 'rb'))
                    for f in os.listdir(pkl_path) if f.endswith('.pkl')), [])

    def _convert_labels_to_indices(self, labels):
        """
        Convert string labels to integer indices

        Args:
            labels: list of string labels or single string label

        Returns:
            Integer indices for the labels
        """
        # Handle single label case
        if isinstance(labels, str):
            labels = [labels]

        # Build label mapping if not exists
        for label in labels:
            if label not in self.label_to_idx:
                idx = len(self.label_to_idx)
                self.label_to_idx[label] = idx
                self.idx_to_label[idx] = label

        # Convert to indices
        indices = [self.label_to_idx[label] for label in labels]

        # Return single index if input was single label
        if len(indices) == 1:
            return indices[0]
        return indices

    def prepare_sample(self, sample):
        # 创建原子图和残基图用于权重计算
        a_weights = Data(
            x=torch.FloatTensor(sample['a_node']),
            edge_index=torch.LongTensor(sample['a_edge_index']),
        )
        r_weights = Data(
            x=torch.FloatTensor(sample['r_node']),
            edge_index=torch.LongTensor(sample['r_edge_index']),
        )

        # 添加高斯边权重
        a_weights = add_gaussian_edge_weights(
            a_weights, sigma=config.GAUSSIAN_SIGMA
        )
        r_weights = add_gaussian_edge_weights(
            r_weights, sigma=config.GAUSSIAN_SIGMA
        )

        # 创建稀疏张量
        atom_adj_t = SparseTensor(
            row=a_weights.edge_index[0],
            col=a_weights.edge_index[1],
            value=a_weights.edge_attr,
            sparse_sizes=(len(a_weights.x), len(a_weights.x))
        ).t()

        residue_adj_t = SparseTensor(
            row=r_weights.edge_index[0],
            col=r_weights.edge_index[1],
            value=r_weights.edge_attr,
            sparse_sizes=(len(r_weights.x), len(r_weights.x))
        ).t()

                # 处理残基级标签 - 每个残基一个标签
        labels = sample['label']
        num_residues = sample['r_node'].shape[0]
        
        if isinstance(labels, str):
            # 标签是字符串格式，每个字符代表一个残基的标签（如 "001011..."）
            if len(labels) != num_residues:
                raise ValueError(f"标签字符串长度({len(labels)})与残基数量({num_residues})不匹配")
            
            # 将字符串转换为整数列表：'0' -> 0, '1' -> 1
            try:
                label_list = [int(char) for char in labels]
                y_tensor = torch.LongTensor(label_list)
            except ValueError as e:
                raise ValueError(f"标签字符串包含非数字字符: {labels[:50]}... 错误: {e}")
                
        elif isinstance(labels, (list, np.ndarray)):
            # 如果标签已经是数组格式
            labels_array = np.array(labels)
            
            if len(labels_array) == 1 and isinstance(labels_array[0], str):
                # 处理包含单个字符串的数组情况
                string_label = labels_array[0]
                if len(string_label) != num_residues:
                    raise ValueError(f"标签字符串长度({len(string_label)})与残基数量({num_residues})不匹配")
                label_list = [int(char) for char in string_label]
                y_tensor = torch.LongTensor(label_list)
            elif labels_array.size != num_residues:
                raise ValueError(f"标签数量({labels_array.size})与残基数量({num_residues})不匹配")
            else:
                # 转换为tensor
                if labels_array.dtype.kind in ['U', 'S', 'O']:  # 字符串类型
                    label_indices = self._convert_labels_to_indices(labels_array.tolist())
                    y_tensor = torch.LongTensor(label_indices)
                else:
                    y_tensor = torch.LongTensor(labels_array.astype(int))
                    
        else:
            raise ValueError(f"不支持的标签类型: {type(labels)}，期望字符串或数组格式")
        
        # 最终检查
        if y_tensor.shape[0] != num_residues:
            raise ValueError(f"最终标签tensor形状({y_tensor.shape})与残基数量({num_residues})不匹配")

        # 构建最终的数据对象
        data = Data(
            atom_x=torch.FloatTensor(sample['a_node']),
            atom_adj_t=atom_adj_t,
            residue_x=torch.FloatTensor(sample['r_node']),
            residue_adj_t=residue_adj_t,
            r_edge_index=torch.LongTensor(sample['r_edge_index']),
            a_edge_index=torch.LongTensor(sample['a_edge_index']),
            y=y_tensor,
            a2r_map=torch.tensor(sample['a2r_map'])
        )

        if self.enable_random_walk_pe:
            residue_transform = T.AddRandomWalkPE(walk_length=self.walk_length, attr_name='r_pe')
            r_pe = Data(
                x=data.residue_x,
                edge_index=data.r_edge_index
            )
            r_pe = residue_transform(r_pe)
            data.r_pe = r_pe.r_pe

        if self.enable_fourier:
            r_fourier = compute_fourier_features(
                data.residue_x,
                data.r_edge_index,
                threshold=self.fourier_threshold
            )

            # 添加傅里叶特征到数据对象
            data.r_fourier = r_fourier

        return data.to(self.device)

    @staticmethod
    def split_data(data, train_ratio=0.8, seed=None):

        if seed is not None:
            np.random.seed(seed)

        data_copy = data.copy()
        np.random.shuffle(data_copy)

        split_index = int(len(data_copy) * train_ratio)
        train_data = data_copy[:split_index]
        val_data = data_copy[split_index:]

        return train_data, val_data

    @staticmethod
    def get_data_info(sample_data):
        return {
            'atom_in_channels': sample_data['a_node'].shape[1],
            'residue_in_channels': sample_data['r_node'].shape[1],
            'atom_nodes': sample_data['a_node'].shape[0],
            'residue_nodes': sample_data['r_node'].shape[0],
            'atom_edges': sample_data['a_edge_index'].shape[1],
            'residue_edges': sample_data['r_edge_index'].shape[1]
        }


def load_data(pkl_path):
    return DataLoader.load_data(pkl_path)


def prepare_sample(sample, device):
    data_loader = DataLoader(device)
    return data_loader.prepare_sample(sample)