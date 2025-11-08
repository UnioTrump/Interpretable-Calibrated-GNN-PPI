"""
Data:
    AAINDEX Data type: torch.FloatTensor, Shape: [N, 566+ 20]
    ESM-C Data type: torch.FloatTensor, Shape: [N, 1125]
    ProtTrans Data type: torch.FloatTensor, Shape: [N, 1024]
"""
import pickle as p
import numpy as np
from torch_geometric.data import Data
import config


class Dataloader:
    def __init__(self):
        self.device = config.DEVICE

    @staticmethod
    def load_data(file_path: list):
        """
        Note:
            file_path is an ordered list: ESM-C, ProtT5, AA, adj, label
        """
        data_list = []
        result_data = []

        for file in file_path:
            with open(file, 'rb') as f:
                data_list.append(p.load(f))

        for i, data_item in enumerate(data_list[0]):  # Use first file as reference
            result_data.append({
                'pid': data_item['PID'],
                'esm_c': data_list[0][i]['x'],
                'prot': data_list[1][i]['x'],
                'AA': data_list[2][i]['AA'],
                'adj': data_list[3][i]['adj'],
                'y': data_list[4][i]['label']
            })

        return result_data

    @staticmethod
    def split_data(All_data, train_ratio=0.8, seed=None):
        np.random.seed(seed)

        data_copy = All_data.copy()
        np.random.shuffle(data_copy)

        split_index = int(len(data_copy) * train_ratio)
        train_data = data_copy[:split_index]
        val_data = data_copy[split_index:]

        return train_data, val_data

    def prepare_sample(self, data_sample):
        data = Data(
            esm=data_sample.get('esm_c', None),
            prot=data_sample.get('prot', None),
            aa=data_sample.get('AA', None),
            adj=data_sample.get('adj', None),
            y=data_sample.get('y', None),
        )
        return data.to(self.device)
