import os
import torch
import numpy as np
from torch_geometric.data import Data
import config
from torch_sparse import SparseTensor

class DataLoader:

    def __init__(self, device=None):
        self.device = device if device is not None else config.DEVICE

    @staticmethod
    def load_data(pkl_path):
        if os.path.isdir(pkl_path):
            all_pkl_files = sorted([f.name for f in os.scandir(pkl_path) if f.is_file() and f.name.endswith('.pkl') and not f.name.startswith('.ipynb_checkpoints')])

            main_pkl_file = os.path.join(pkl_path, all_pkl_files[0])
        elif os.path.isfile(pkl_path) and pkl_path.endswith('.pkl'):
            main_pkl_file = pkl_path
        else:
            raise ValueError(f"Invalid pkl_path: {pkl_path}. Must be a directory containing .pkl files or a .pkl file.")

        with open(main_pkl_file, 'rb') as f:
            return torch.load(f)

    def prepare_sample(self, sample_data):
        y=torch.LongTensor(sample_data['y'])
        data = Data(
            x=sample_data.get('r_node', None),
            adj_t=sample_data.get('residue_adj_t', None),
            pe=sample_data.get('r_pe', None),
            fourier=sample_data.get('r_fourier', None),
            y=y
        )
        return data.to(self.device)

    @staticmethod
    def split_data(all_data_samples, train_ratio=0.8, seed=None):
        if seed is not None:
            np.random.seed(seed)

        data_copy = all_data_samples.copy()
        np.random.shuffle(data_copy)

        split_index = int(len(data_copy) * train_ratio)
        train_data = data_copy[:split_index]
        val_data = data_copy[split_index:]

        return train_data, val_data
