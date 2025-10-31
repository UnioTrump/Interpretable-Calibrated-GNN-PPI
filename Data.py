"""
AAINDEX+BLOSUM62 Data type: torch.FloatTensor, Shape: [N, 566+ 20]
ESM-C Data type: torch.FloatTensor, Shape: [N, 2560]
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
        file_path is an ordered list: ESM-C, ProtT5, B_A, adj, label
        """
        data_list = []
        for file in file_path:
            with open(file, 'rb') as f:
                data = p.load(f)
            data_list.append(data)

        Data=[]
        for pid in data:
            data= {'pid': pid, 'esm_c': data_list[0][pid], 'prot': data_list[1][pid],
                   'b_a': data_list[2][pid], 'adj':data_list[3][pid], 'y': data_list[4][pid]}
            Data.append(data)
        return Data

    def prepare_sample(self, data_sample):
        data = Data(
            esm=data_sample.get('esm_c', None),
            prot=data_sample.get('prot', None),
            b_a=data_sample.get('b_a', None),
            adj=data_sample.get('adj', None),
            y=data_sample.get('y', None),
        )
        return data.to(self.device)

    @staticmethod
    def split_data(All_data, train_ratio=0.8, seed=None):
        np.random.seed(seed)

        data_copy = All_data.copy()
        np.random.shuffle(data_copy)

        split_index = int(len(data_copy) * train_ratio)
        train_data = data_copy[:split_index]
        val_data = data_copy[split_index:]

        return train_data, val_data