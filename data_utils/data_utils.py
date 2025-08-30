import pickle
import torch
import os
import numpy as np
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from .utils import add_gaussian_edge_weights
import torch_geometric.transforms as T
import config


def frequency_filtering(eigenvalues, x_low, x_high):

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

    num_nodes = x.shape[0]
    device = x.device

    # 构建邻接矩阵
    edge_adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float, device=device)
    edge_adj[edge_index[0, :], edge_index[1, :]] = 1

    # 图傅里叶变换
    adj = edge_adj + edge_adj.t()  # 确保对称
    # adj = adj.fill_diagonal_(0)  # 去除自环

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

    def __init__(self, device=None, enable_fourier=True, fourier_threshold=config.FOURIER_THRESHOLD, enable_random_walk_pe=True,
                 walk_length=config.PE_DIM):
        self.device = config.DEVICE
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

        labels = sample['label']
        label_list = [int(char) for char in labels]
        y_tensor = torch.LongTensor(label_list)

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
            num_nodes = data.r_pe.shape[0]
            r_fourier = r_fourier.view(num_nodes, num_nodes)
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