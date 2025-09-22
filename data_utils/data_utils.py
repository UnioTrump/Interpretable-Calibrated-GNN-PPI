import os
import torch
import numpy as np
from torch_geometric.data import Data
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


def compute_fourier_features(x, edge_index):

    num_nodes = x.shape[0]
    device = x.device

    # 构建邻接矩阵
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float64, device=device)
    adj[edge_index[0, :], edge_index[1, :]] = 1

    # 图傅里叶变换
    adj = adj + adj.t()  # 确保对称
    adj = adj.fill_diagonal_(0)  # 去除自环！！！！！一定要去除自环

    # 计算度矩阵
    degree = torch.diag(adj.sum(dim=1))
    if torch.any(torch.diag(degree) == 0):
        # print("Waning ! There is a gulijiedian")
        epsilon = 1e-5
        eye_matrix = epsilon * torch.eye(degree.shape[0], device=device)
        d_sqrt = torch.inverse(torch.sqrt(degree + eye_matrix))
    else:
        d_sqrt = torch.inverse(torch.sqrt(degree))

    # 计算归一化的拉普拉斯矩阵
    laplacian = torch.eye(num_nodes, device=device) - d_sqrt @ adj @ d_sqrt

    # 进行特征分解
    eigvals, eigvecs = torch.linalg.eig(laplacian)
    # 取实部
    eigvals = eigvals.real.float()
    eigvecs = eigvecs.real.float()
    # Graph傅里叶变换
    x_fourier = eigvecs.t() @ x.float()

    eigvals, idx = torch.sort(eigvals)
    eigvecs = eigvecs[:, idx]
    nodes = eigvecs.shape[0]
    k=int(0.2 * nodes)

    # 分离低频和高频信号
    low_mask = torch.zeros_like(eigvals)
    low_mask[:k] = 1
    high_mask = 1 - low_mask

    x_low = eigvecs @ (x_fourier * low_mask.unsqueeze(1))
    x_high = eigvecs @ (x_fourier * high_mask.unsqueeze(1))

    # 生成注意力优化矩阵
    attention_optimization_matrix = frequency_filtering(eigvals, x_low, x_high)

    return attention_optimization_matrix

class DataLoader:

    def __init__(self, device = None):
        self.device = config.DEVICE

    @staticmethod
    def load_data(pkl_path):
        return sum((torch.load(open(os.path.join(pkl_path, f), 'rb'))
                    for f in os.listdir(pkl_path) if f.endswith('.pkl')), [])

    def prepare_sample(self, sample):

        data = Data(
            seq_x=sample['r_node'],
            seq_adj_t=sample['residue_adj_t'],
            #r_edge_index=sample['r_edge_index'],
            y=sample['y'],
            r_pe=sample['r_pe'],
            r_fourier=sample['r_fourier']
        )

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
            'sequence_in_channels': sample_data['r_node'].shape[1],
            'sequence_nodes': sample_data['r_node'].shape[0],
        }

def load_data(pkl_path):
    return DataLoader.load_data(pkl_path)


def prepare_sample(sample, device):
    data_loader = DataLoader(device)
    return data_loader.prepare_sample(sample)
