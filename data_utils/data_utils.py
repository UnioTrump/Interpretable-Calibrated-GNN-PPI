import os
import torch
import numpy as np
from torch_geometric.data import Data
import config
from torch_sparse import SparseTensor

def frequency_filtering(eigenvalues, x_low, x_high):

    num_nodes = x_low.shape[0]
    eigenvalues_reshaped_i = eigenvalues.view(-1, 1)
    eigenvalues_reshaped_j = eigenvalues.view(1, -1)
    sum_matrix = eigenvalues_reshaped_i + eigenvalues_reshaped_j

    low_energy = torch.sum(x_low ** 2, dim=1)
    high_energy = torch.sum(x_high ** 2, dim=1)

    low_energy_reshaped_i = low_energy.view(-1, 1)
    high_energy_reshaped_j = high_energy.view(1, -1)
    denominator = low_energy.sum() + high_energy.sum()

    if denominator == 0:
        denominator = 1e-8

    filter_matrix = (low_energy_reshaped_i + high_energy_reshaped_j) / denominator

    attention_optimization_matrix = sum_matrix * filter_matrix

    attention_optimization_matrix = torch.nan_to_num(attention_optimization_matrix, nan=0.0)
    return attention_optimization_matrix


def compute_fourier_features(x, edge_index):

    num_nodes = x.shape[0]
    device = x.device

    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float64, device=device)
    adj[edge_index[0, :], edge_index[1, :]] = 1

    adj = adj + adj.t()
    adj = adj.fill_diagonal_(0)

    degree = torch.diag(adj.sum(dim=1))
    if torch.any(torch.diag(degree) == 0):
        epsilon = 1e-5
        eye_matrix = epsilon * torch.eye(degree.shape[0], device=device)
        d_sqrt = torch.inverse(torch.sqrt(degree + eye_matrix))
    else:
        d_sqrt = torch.inverse(torch.sqrt(degree))

    laplacian = torch.eye(num_nodes, device=device) - d_sqrt @ adj @ d_sqrt

    eigvals, eigvecs = torch.linalg.eig(laplacian)
    eigvals = eigvals.real.float()
    eigvecs = eigvecs.real.float()
    x_fourier = eigvecs.t() @ x.float()

    eigvals, idx = torch.sort(eigvals)
    eigvecs = eigvecs[:, idx]
    nodes = eigvecs.shape[0]
    k=int(0.2 * nodes)

    low_mask = torch.zeros_like(eigvals)
    low_mask[:k] = 1
    high_mask = 1 - low_mask

    x_low = eigvecs @ (x_fourier * low_mask.unsqueeze(1))
    x_high = eigvecs @ (x_fourier * high_mask.unsqueeze(1))

    attention_optimization_matrix = frequency_filtering(eigvals, x_low, x_high)

    return attention_optimization_matrix

class DataLoader:

    def __init__(self, device=None, multimodal_data_dir=None):
        self.device = device if device is not None else config.DEVICE

        self.all_modal_data_lists = []
        self.modal_names_list = []

        if multimodal_data_dir:
            all_pkl_files = sorted([f.name for f in os.scandir(multimodal_data_dir) if f.is_file() and f.name.endswith('.pkl') and not f.name.startswith('.ipynb_checkpoints')])

            if not all_pkl_files:
                raise FileNotFoundError(f"No .pkl files found in {multimodal_data_dir}")

            for pkl_file_name in all_pkl_files:
                modal_name = os.path.splitext(pkl_file_name)[0]
                modal_path = os.path.join(multimodal_data_dir, pkl_file_name)
                with open(modal_path, 'rb') as f:
                    self.all_modal_data_lists.append(torch.load(f))
                    self.modal_names_list.append(modal_name)

            if self.all_modal_data_lists:
                first_modal_len = len(self.all_modal_data_lists[0])
                for i, modal_list in enumerate(self.all_modal_data_lists):
                    if len(modal_list) != first_modal_len:
                        raise ValueError(f"Multimodal list at index 0 has length {first_modal_len}, but list at index {i} ('{self.modal_names_list[i]}') has length {len(modal_list)}. All multimodal lists must have the same length.")

    @staticmethod
    def load_data(data_loader_instance):
        if data_loader_instance.all_modal_data_lists:
            return list(range(len(data_loader_instance.all_modal_data_lists[0])))
        else:
            return []

    def prepare_sample(self, idx):
        def move_to_device(dat_item, target_device):
            if isinstance(dat_item, torch.Tensor):
                return dat_item.to(target_device)
            elif isinstance(dat_item, SparseTensor):
                return dat_item.to(target_device)
            return dat_item

        all_modal_sample_data = {}

        for i, modal_list in enumerate(self.all_modal_data_lists):
            modal_name = self.modal_names_list[i]
            sample_modal = modal_list[idx]
            all_modal_sample_data[modal_name] = {
                'x': move_to_device(sample_modal.get('r_node', None), self.device),
                'adj_t': move_to_device(sample_modal.get('residue_adj_t', None), self.device),
                'pe': move_to_device(sample_modal.get('r_pe', None), self.device),
                'fourier': move_to_device(sample_modal.get('r_fourier', None), self.device),
                'y': move_to_device(sample_modal.get('y', None), self.device)
            }

        y_label = None
        if self.modal_names_list and self.modal_names_list[0] in all_modal_sample_data:
            y_label = all_modal_sample_data[self.modal_names_list[0]].get('y', None)

        data = Data(y=y_label)

        for modal_name, modal_data in all_modal_sample_data.items():
            setattr(data, f'{modal_name}_x', modal_data['x'])
            setattr(data, f'{modal_name}_adj_t', modal_data['adj_t'])
            setattr(data, f'{modal_name}_pe', modal_data['pe'])
            setattr(data, f'{modal_name}_fourier', modal_data['fourier'])

        data.modal_names_list = self.modal_names_list

        return data

    @staticmethod
    def split_data(dat_indices, train_ratio=0.8, seed=None):
        if seed is not None:
            np.random.seed(seed)

        data_copy = dat_indices.copy()
        np.random.shuffle(data_copy)

        split_index = int(len(data_copy) * train_ratio)
        train_data = data_copy[:split_index]
        val_data = data_copy[split_index:]

        return train_data, val_data

    @staticmethod
    def dat_ifo(sample_data):
        info = {}

        if not hasattr(sample_data, 'modal_names_list') or not sample_data.modal_names_list:
            return info

        for modal_name in sample_data.modal_names_list:
            x_attr = getattr(sample_data, f'{modal_name}_x', None)
            pe_attr = getattr(sample_data, f'{modal_name}_pe', None)
            fourier_attr = getattr(sample_data, f'{modal_name}_fourier', None)

            if x_attr is not None:
                info[f'{modal_name}_in_channels'] = x_attr.shape[1]
                info[f'{modal_name}_nodes'] = x_attr.shape[0]
            if pe_attr is not None:
                info[f'{modal_name}_pe_dim'] = pe_attr.shape[1]
            if fourier_attr is not None:
                info[f'{modal_name}_fourier_dim'] = fourier_attr.shape[1]
        return info
