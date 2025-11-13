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
        self.device = 'cpu'

    @staticmethod
    def load_data(file_path: list):
        """
        Note:
            file_path is an ordered list: [ESM-C, ProtT5, AA, adj, label]
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

    def prepare_sample(self, data_sample, ratio=2):

        pid = data_sample["pid"]
        aa = data_sample["AA"]
        esmc = data_sample["esm_c"]
        prot = data_sample["prot"]
        adj = data_sample["adj"]
        labels = data_sample["y"]

        labels_np = labels.cpu().numpy() if hasattr(labels, 'cpu') else labels
        ppi_idx = np.where(labels_np == 1)[0]
        non_ppi_idx = np.where(labels_np == 0)[0]
        Nr_p, Nr_n = len(ppi_idx), len(non_ppi_idx)

        M = int(np.ceil(Nr_n / (ratio * Nr_p)))
        print(f'M: {M}\nPos: {Nr_p}\nNeg: {Nr_n}')
        np.random.shuffle(non_ppi_idx)

        subsets = []
        for i in range(M):
            start = i * int(ratio * Nr_p)
            end = min((i + 1) * int(ratio * Nr_p), Nr_n)
            non_ppi_part = non_ppi_idx[start:end]

            subset_idx = np.concatenate([ppi_idx, non_ppi_part])
            np.random.shuffle(subset_idx)

            subsets.append(Data(
                pid=pid,
                esm=esmc[subset_idx],
                prot=prot[subset_idx],
                aa=aa[subset_idx],
                adj=adj[subset_idx][:, subset_idx],
                y=labels[subset_idx],
            ).to(self.device))

        return subsets

if __name__ == '__main__':
    dalo=Dataloader()
    dl = dalo.load_data(config.VAL1)
    for d in dl:
        sub = dalo.prepare_sample(data_sample=d)
        break

    labels = sub[0].y
    ppi_idx = np.where(labels == 1)[0]
    non_ppi_idx = np.where(labels == 0)[0]
    Nr_p, Nr_n = len(ppi_idx), len(non_ppi_idx)
    print(f'Pos: {Nr_p}, Neg: {Nr_n}')