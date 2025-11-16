"""
Data:
    AAINDEX Data type: torch.FloatTensor, Shape: [N, 566+ 20]
    ESM-C Data type: torch.FloatTensor, Shape: [N, 1125]
    ProtTrans Data type: torch.FloatTensor, Shape: [N, 1024]
"""
from tqdm import tqdm
import pickle as p
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import config
from torch_sparse import SparseTensor
device = 'cpu'

class PPIDataset(Dataset):
    """
    PyTorch Dataset for PPI data.
    Each item is a dict with keys: pid, esm_c, prot, AA, adj, y
    The 'adj' field will be converted to torch_sparse.SparseTensor.
    """
    def __init__(self, data_list, is_training, sample_ratio=2):
        self.data_list = data_list
        self.sample_ratio = sample_ratio
        self.is_training = is_training
        self.device = device
        self.samples = []
        self._prepare()
    def _prepare(self):
        if self.is_training:
            self._prepare_samples()
        else:
            self._prepare_val()
    def _prepare_samples(self):

        # For training, do positive/negative sampling
        for d in tqdm(self.data_list, total=len(self.data_list)):
            try:
                labels = d['y'].detach().clone()
                ppi_idx = (labels == 1).nonzero(as_tuple=True)[0]
                non_ppi_idx = (labels == 0).nonzero(as_tuple=True)[0]
                Nr_p, Nr_n = len(ppi_idx), len(non_ppi_idx)

                M = int(np.ceil(Nr_n / (self.sample_ratio * Nr_p))) if Nr_n > 0 else 1
                non_ppi_idx = non_ppi_idx[torch.randperm(Nr_n)] if Nr_n > 0 else non_ppi_idx
                for i in range(M):
                    start = i * int(self.sample_ratio * Nr_p)
                    end = min((i + 1) * int(self.sample_ratio * Nr_p), Nr_n) if Nr_n > 0 else 0
                    non_ppi_part = non_ppi_idx[start:end] if Nr_n > 0 else torch.tensor([], dtype=torch.long)
                    # build subset indices safely (handle empty non_ppi_part)
                    if non_ppi_part.numel() > 0:
                        subset_idx = torch.cat([ppi_idx, non_ppi_part])
                    else:
                        subset_idx = ppi_idx.clone()
                    # shuffle subset indices
                    subset_idx = subset_idx[torch.randperm(subset_idx.size(0))]

                    # slice arrays/tensors for this subset
                    esm_slice = d['esm_c'][subset_idx]
                    prot_slice = d['prot'][subset_idx]
                    AA_slice = d['AA'][subset_idx]

                    row, col, val = d['adj'].coo()
                    # print((col))
                    # keep edges where both endpoints are in subset_idx
                    mask_row = torch.zeros(d['adj'].sparse_sizes()[0], dtype=torch.bool, device=self.device)
                    mask_row[subset_idx.to(self.device)] = True
                    # select edges
                    keep = mask_row[row] & mask_row[col]
                    new_row = row[keep]
                    new_col = col[keep]
                    new_val = val[keep]
                    # remap indices to new range 0..len(subset_idx)-1
                    # create mapping
                    inv_idx = -torch.ones(d['adj'].sparse_sizes()[0], dtype=torch.long, device=self.device)
                    inv_idx[subset_idx.to(self.device)] = torch.arange(len(subset_idx), device=self.device)
                    if new_row.numel() > 0:
                        new_row = inv_idx[new_row]
                        new_col = inv_idx[new_col]
                    adj_slice = SparseTensor(row=new_row, col=new_col, value=new_val, sparse_sizes=(len(subset_idx), len(subset_idx)))
                    # print(adj_slice)
                    self.samples.append({
                        'pid': d['pid'],
                        'esm_c': esm_slice,
                        'prot': prot_slice,
                        'AA': AA_slice,
                        'adj': adj_slice,
                        'y': labels[subset_idx]
                    })

            except Exception as e:
                # print(e)
                continue

    def _prepare_val(self):
        # For validation/testing, use full data
        for d in tqdm(self.data_list, total=len(self.data_list)):
            try:
                if len(d['y']) != d['adj'].size(0):
                    continue
                self.samples.append({
                    'pid': d['pid'],
                    'esm_c': d['esm_c'],
                    'prot': d['prot'],
                    'AA': d['AA'],
                    'adj': d['adj'],
                    'y': d['y']
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
    """
    Custom collate for batching samples that include a torch_sparse.SparseTensor 'adj'.
    Produces a single batched adjacency by offsetting node indices so each sample becomes
    a block on the diagonal of the big adjacency matrix.
    """

    AA_list, esm_list, prot_list, y_list = [], [], [], []
    rows, cols, edge_attrs = [], [], []
    pid_list = []

    node_offset = 0
    for s in _batch:
        AA = s['AA']
        esm = s['esm_c']
        prot = s['prot']
        y = s['y']
        adj = s['adj']

        n_nodes = AA.size(0)
        AA_list.append(AA)
        esm_list.append(esm)
        prot_list.append(prot)
        y_list.append(y)
        pid_list.append(s.get('pid', None))

        # adj expected to be torch_sparse.SparseTensor
        row, col, edge_attr = adj.coo()

        rows.append(row + node_offset)
        cols.append(col + node_offset)
        edge_attrs.append(edge_attr)

        node_offset += n_nodes

    AA_batch = torch.cat(AA_list, dim=0)
    esm_batch = torch.cat(esm_list, dim=0)
    prot_batch = torch.cat(prot_list, dim=0)
    y_batch = torch.cat(y_list, dim=0)

    row_all = torch.cat(rows, dim=0)
    col_all = torch.cat(cols, dim=0)
    edge_attr_all = torch.cat(edge_attrs, dim=0)


    adj_batch = SparseTensor(row=row_all, col=col_all, value=edge_attr_all, sparse_sizes=(node_offset, node_offset))

    return {
        'pid': pid_list,
        'esm_c': esm_batch,
        'prot': prot_batch,
        'AA': AA_batch,
        'adj': adj_batch,
        'y': y_batch
    }


class PPIData:
    def __init__(self):
        self.device = config.DEVICE

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

        for i, data_item in enumerate(data_list[0]):  # Use the first file as reference
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
        train_d = data_copy[:split_index]
        val_d = data_copy[split_index:]

        return train_d, val_d

if __name__ == '__main__':

    dl = PPIData.load_data(config.VAL1)
    train_data, val_data = PPIData.split_data(dl[:10])

    train_dataset = PPIDataset(train_data, sample_ratio=2, is_training=True)
    val_dataset = PPIDataset(val_data, sample_ratio=2, is_training=False)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=sparse_collate)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=sparse_collate)
    for batch in train_loader:
        print(batch['pid'], batch['y'].shape)
    for batch in val_loader:
        print(batch['pid'], batch['y'].shape)
