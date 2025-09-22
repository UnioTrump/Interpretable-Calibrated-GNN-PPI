import os
import torch
import numpy as np
from torch_geometric.data import Data
import config
from torch_sparse import SparseTensor # Import SparseTensor

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

    def __init__(self, device=None, multimodal_data_dir=None):
        self.device = device if device is not None else config.DEVICE

        self.modal1_list = []
        self.modal2_list = []
        self.modal3_list = []

        if multimodal_data_dir:
            all_pkl_files = sorted([f for f in os.listdir(multimodal_data_dir) if f.endswith('.pkl')])
            if len(all_pkl_files) != 3:
                raise ValueError(f"Expected 3 pkl files in {multimodal_data_dir}, but found {len(all_pkl_files)}")
            
            with open(os.path.join(multimodal_data_dir, all_pkl_files[0]), 'rb') as f:
                self.modal1_list = torch.load(f)
            with open(os.path.join(multimodal_data_dir, all_pkl_files[1]), 'rb') as f:
                self.modal2_list = torch.load(f)
            with open(os.path.join(multimodal_data_dir, all_pkl_files[2]), 'rb') as f:
                self.modal3_list = torch.load(f)

        # Ensure all loaded modal lists have the same length
        if self.modal1_list and self.modal2_list and self.modal3_list:
            assert len(self.modal1_list) == len(self.modal2_list) == len(self.modal3_list), \
                "All multimodal lists must have the same length."
        elif self.modal1_list or self.modal2_list or self.modal3_list:
            lengths = [len(l) for l in [self.modal1_list, self.modal2_list, self.modal3_list] if l]
            if len(set(lengths)) > 1:
                raise ValueError("Loaded multimodal lists have inconsistent lengths.")

    @staticmethod
    def load_data(data_loader_instance): # Removed pkl_path parameter
        if data_loader_instance.modal1_list:
            return list(range(len(data_loader_instance.modal1_list)))
        else:
            return [] # No data loaded

    def prepare_sample(self, idx):
        # Get data for each modality using the index
        sample_modal1 = self.modal1_list[idx] if self.modal1_list else {}
        sample_modal2 = self.modal2_list[idx] if self.modal2_list else {}
        sample_modal3 = self.modal3_list[idx] if self.modal3_list else {}
        
        # Helper function to move tensor to device, handle SparseTensor
        def move_to_device(dat_item, target_device):
            if isinstance(dat_item, torch.Tensor):
                return dat_item.to(target_device)
            elif isinstance(dat_item, SparseTensor):
                return dat_item.to(target_device)
            return dat_item # Return as is if not a tensor/sparse tensor


        # Original main data fields are assumed to be in sample_modal1
        seq_x = move_to_device(sample_modal1.get('r_node', None), self.device)
        seq_adj_t = move_to_device(sample_modal1.get('residue_adj_t', None), self.device)
        y = move_to_device(sample_modal1.get('y', None), self.device)
        r_pe = move_to_device(sample_modal1.get('r_pe', None), self.device)
        r_fourier = move_to_device(sample_modal1.get('r_fourier', None), self.device)


        # Create the Data object
        data = Data(
            seq_x=seq_x,
            seq_adj_t=seq_adj_t,
            y=y,
            r_pe=r_pe,
            r_fourier=r_fourier
        )

        # Add additional multimodal data to the Data object
        if sample_modal2:
            data.modal2_x = move_to_device(sample_modal2.get('r_node', None), self.device)
            data.modal2_adj_t = move_to_device(sample_modal2.get('residue_adj_t', None), self.device)
            data.modal2_r_pe = move_to_device(sample_modal2.get('r_pe', None), self.device)
            data.modal2_r_fourier = move_to_device(sample_modal2.get('r_fourier', None), self.device)

        if sample_modal3:
            data.modal3_x = move_to_device(sample_modal3.get('r_node', None), self.device)
            data.modal3_adj_t = move_to_device(sample_modal3.get('residue_adj_t', None), self.device)
            data.modal3_r_pe = move_to_device(sample_modal3.get('r_pe', None), self.device)
            data.modal3_r_fourier = move_to_device(sample_modal3.get('r_fourier', None), self.device)

        return data

    @staticmethod
    def split_data(dat_indices, train_ratio=0.8, seed=None): # data is now a list of indices
        if seed is not None:
            np.random.seed(seed)

        data_copy = dat_indices.copy() # Operate on indices
        np.random.shuffle(data_copy)

        split_index = int(len(data_copy) * train_ratio)
        train_data = data_copy[:split_index]
        val_data = data_copy[split_index:]

        return train_data, val_data

    @staticmethod
    def data_ifo(sample_data): # sample_data will now be a Data object from prepare_sample
        info = {
            'sequence_in_channels': sample_data.seq_x.shape[1],
            'sequence_nodes': sample_data.seq_x.shape[0],
        }
        if hasattr(sample_data, 'modal2_x'):
            info['modal2_in_channels'] = sample_data.modal2_x.shape[1]
        if hasattr(sample_data, 'modal3_x'):
            info['modal3_in_channels'] = sample_data.modal3_x.shape[1]
        if hasattr(sample_data, 'modal2_r_pe'):
            info['modal2_pe_dim'] = sample_data.modal2_r_pe.shape[1]
        if hasattr(sample_data, 'modal3_r_pe'):
            info['modal3_pe_dim'] = sample_data.modal3_r_pe.shape[1]
        return info
