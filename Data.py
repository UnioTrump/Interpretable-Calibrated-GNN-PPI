"""
Data:
    AAINDEX Data type: torch.FloatTensor, Shape: [N, 566]
    DSSP Data type: torch.FloatTensor, Shape: [N, 14]
    BLOSUM62 Data type: torch.FloatTensor, Shape: [N, 20]
    ESM-C Data type: torch.FloatTensor, Shape: [N, 1125]
"""
import os
from tqdm import tqdm
import pickle as p
import numpy as np
import torch
from torch.utils.data import Dataset
import config
from torch_sparse import SparseTensor
device = 'cpu'

class PPIDataset(Dataset):

    def __init__(self, data_list):
        self.data_list = data_list
        self.device = device
        self.samples = []
        self._prepare_val()

    def _prepare_val(self):

        for d in tqdm(self.data_list, total=len(self.data_list)):
            try:
                if len(d['y']) != d['adj'].size(0):
                    continue
                self.samples.append({
                    'pid': d['pid'],
                    'esm_c': d['esm_c'],
                    'AA': d['AA'],
                    'BLOSUM': d['BLOSUM'],
                    'dssp': d['dssp'],
                    'adj': d['adj'],
                    'y': d['y'],
                    'pse': d['pse'],
                    'res_atom': d['res_atom'],
                })
            except Exception as e:
                print(e)
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample


def sparse_collate(_batch):

    AA_list, esm_list, BLOSUM_list, dssp_list, y_list, pse_list, res_atom_list = [], [], [], [], [], [], []
    rows, cols, edge_attrs = [], [], []
    pid_list = []

    node_offset = 0
    for s in _batch:
        AA = s['AA']
        esm = s['esm_c']
        BLOSUM = s['BLOSUM']
        dssp = s['dssp']
        y = s['y']
        adj = s['adj']
        pse = s['pse']
        res_atom = s['res_atom']

        n_nodes = AA.size(0)
        AA_list.append(AA)
        esm_list.append(esm)
        BLOSUM_list.append(BLOSUM)
        dssp_list.append(dssp)
        y_list.append(y)
        pid_list.append(s.get('pid', None))
        pse_list.append(pse)
        res_atom_list.append(res_atom)

        row, col, edge_attr = adj.coo()
        rows.append(row + node_offset)
        cols.append(col + node_offset)
        edge_attrs.append(edge_attr)
        node_offset += n_nodes

    AA_batch = torch.cat(AA_list, dim=0)
    esm_batch = torch.cat(esm_list, dim=0)
    BLOSUM_batch = torch.cat(BLOSUM_list, dim=0)
    dssp_batch = torch.cat(dssp_list, dim=0)
    y_batch = torch.cat(y_list, dim=0)
    pse_batch = torch.cat(pse_list, dim=0)
    res_atom_batch = torch.cat(res_atom_list, dim=0)

    row_all = torch.cat(rows, dim=0)
    col_all = torch.cat(cols, dim=0)
    edge_attr_all = torch.cat(edge_attrs, dim=0)

    adj_batch = SparseTensor(row=row_all, col=col_all, value=edge_attr_all, sparse_sizes=(node_offset, node_offset))

    return {
        'pid': pid_list,
        'esm_c': esm_batch,
        'AA': AA_batch,
        'BLOSUM': BLOSUM_batch,
        'dssp': dssp_batch,
        'adj': adj_batch,
        'y': y_batch,
        'pse': pse_batch,
        'res_atom': res_atom_batch,
    }


class PPIData:
    def __init__(self):
        self.device = config.DEVICE

    @staticmethod
    def load_data(folder_path: str):
        """
        Note:
            There are ordered: aaindex, BLOSUM, dssp, edge, ESM, label
        """
        file_keywords = ['aaindex', 'BLOSUM', 'dssp', 'edge', 'ESMC', 'label', 'pse_1a','res_atom_1a']
        all_files = os.listdir(folder_path)
        pkl_files = [f for f in all_files if f.endswith('.pkl')]

        ordered_files = []
        for key in file_keywords:
            matched = [f for f in pkl_files if key.lower() in f.lower()]
            if not matched:
                raise FileNotFoundError(f"No file with keyword '{key}' found in {folder_path}")
            ordered_files.append(os.path.join(folder_path, matched[0]))

        data_list = []
        for file_path in ordered_files:
            with open(file_path, 'rb') as f:
                data_list.append(p.load(f))

        result_data = []
        for i, _ in enumerate(data_list[0]):
            result_data.append({
                'pid': data_list[0][i]['PID'],
                'AA': data_list[0][i]['AA'],
                'BLOSUM': data_list[1][i]['x'],
                'dssp': data_list[2][i]['x'],
                'adj': data_list[3][i]['adj'],
                'esm_c': data_list[4][i]['x'],
                'y': data_list[5][i]['label'],
                'pse': data_list[6][i]['x'],
                'res_atom': data_list[7][i]['x'],
            })
        return result_data

    @staticmethod
    def split_data(All_data, train_ratio=0.8, test_ratio=0.1, seed=None):

        if train_ratio + test_ratio > 1.0:
            raise ValueError("train_ratio + test_ratio must be <= 1.0")

        np.random.seed(seed)
        data_copy = list(All_data)
        np.random.shuffle(data_copy)

        n_total = len(data_copy)
        train_end = int(n_total * train_ratio)
        test_end = train_end + int(n_total * test_ratio)

        train_d = data_copy[:train_end]
        test_d = data_copy[train_end:test_end]
        val_d = data_copy[test_end:]

        assert len(train_d) + len(test_d) + len(val_d) == n_total

        if test_ratio == 0:
            return train_d, val_d

        return train_d, test_d, val_d
