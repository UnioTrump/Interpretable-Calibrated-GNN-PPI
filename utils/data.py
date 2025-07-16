import pickle
import os
import torch
from torch_geometric.data import InMemoryDataset, Data
import tqdm

class ProteinGraphDataset(InMemoryDataset):
    def __init__(self, root, data_path, transform=None, pre_transform=None):
        self.data_path = data_path
        super(ProteinGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.path.basename(self.data_path)]

    @property
    def processed_file_names(self):
        return 'processed_data.pt'

    def download(self):
        pass

    def process(self):
        with open(self.data_path, 'rb') as f:
            raw_data_list = pickle.load(f)

        data_list = []
        pbar = tqdm.tqdm(raw_data_list, desc="Processing proteins")
        for i, protein_data in enumerate(pbar):
            atom_x = torch.tensor(protein_data['atom_graph_node'], dtype=torch.float)
            atom_edge_index = torch.tensor(protein_data['atom_graph_edge'], dtype=torch.long)
            residue_x = torch.tensor(protein_data['residue_graph_node'], dtype=torch.float)
            residue_edge_index = torch.tensor(protein_data['residue_graph_edge'], dtype=torch.long)
            a2r_map = torch.tensor(protein_data['a2r_map'], dtype=torch.long)
            y = torch.tensor(protein_data['label'], dtype=torch.float).squeeze()

            # Ensure y is a scalar tensor for single-value labels
            if y.dim() == 0:
                y = y.unsqueeze(0)

            data = Data(
                atom_x=atom_x,
                atom_edge_index=atom_edge_index,
                residue_x=residue_x,
                residue_edge_index=residue_edge_index,
                a2r_map=a2r_map,
                y=y,
                id=i
            )
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_paths[0])
        return data[idx]

    def __len__(self):
        data, slices = torch.load(self.processed_paths[0])
        return len(slices['y'])

    def __getitem__(self, idx):
        return self.get(idx)

def load_dataset(data_path, **kwargs):
    dataset = ProteinGraphDataset(root='data', data_path=data_path, **kwargs)
    return dataset
