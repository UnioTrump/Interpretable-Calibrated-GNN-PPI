from tqdm import tqdm
import random
from torch_geometric.data import Data, Dataset, Batch
import torch
# from config import DefaultConfig
from torch.utils.data import DataLoader  # 改为使用 PyTorch 原生的 DataLoader
from torch_sparse import SparseTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ProteinData(Dataset):
    def __init__(self, raw_dict):
        super().__init__()
        self.samples = []
        # Verify feature dimensions are consistent
        sample_atom_dim = len(raw_dict[0]['atom_graph_node'][0])
        sample_residue_dim = len(raw_dict[0]['residue_graph_node'][0])

        print(f"Using atom feature dimension: {sample_atom_dim}")
        print(f"Using residue feature dimension: {sample_residue_dim}")

        for train_p in raw_dict:
            # Convert features directly to tensors without padding
            train_p_a_node = torch.FloatTensor(train_p['atom_graph_node']).to(device)
            train_p_r_node = torch.FloatTensor(train_p['residue_graph_node']).to(device)

            train_p_a_edge = torch.LongTensor(train_p['atom_graph_edge']).to(device)
            train_p_r_edge = torch.LongTensor(train_p['residue_graph_edge']).to(device)
            train_p_label = torch.LongTensor(train_p['label']).to(device)
            train_p_ar_map = torch.tensor(train_p['a2r_map']).to(device)

            # Atom graph
            atom_graph = Data(
                x=train_p_a_node,
                batch=torch.zeros(train_p_a_node.size(0), dtype=torch.long).to(device),
                mark=torch.zeros(train_p_a_node.size(0), dtype=torch.long).to(device),
                edge_index=train_p_a_edge,
                edge_attr=torch.ones(train_p_a_edge.shape[1], dtype=torch.float).to(device)
            )

            # Residue graph
            residue_graph = Data(
                x=train_p_r_node,
                batch=torch.zeros(train_p_r_node.size(0), dtype=torch.long).to(device),
                mark=torch.zeros(train_p_r_node.size(0), dtype=torch.long).to(device),
                edge_index=train_p_r_edge,
                edge_attr=torch.ones(train_p_r_edge.shape[1], dtype=torch.float).to(device)
            )

            self.samples.append({
                'atom_graph': atom_graph,
                'residue_graph': residue_graph,
                'a2r_map': train_p_ar_map,
                'label': train_p_label,
            })

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            new_instance = ProteinData.__new__(ProteinData)
            new_instance.samples = self.samples[idx]
            # Initialize other necessary attributes to avoid AttributeError
            new_instance.__init__ = self.__init__.__func__
            return new_instance
        return self.samples[idx]

    def get(self, idx):
        return self.__getitem__(idx)

    def __setitem__(self, idx, value):
        self.samples[idx] = value

    def __len__(self):
        return len(self.samples)

    def len(self):
        return self.__len__()

    def shuffle(self):
        """Safely shuffle the data"""
        random.shuffle(self.samples)

    @property
    def atom_graphs(self):
        """Dynamically generate atom_graph list to avoid manual synchronization"""
        return [sample["atom_graph"] for sample in self.samples]

    @property
    def residue_graphs(self):
        """Dynamically generate residue_graph list"""
        return [sample["residue_graph"] for sample in self.samples]

    @property
    def a2r_maps(self):
        """Dynamically generate a2r_map list"""
        return [sample["a2r_map"] for sample in self.samples]

    @property
    def labels(self):
        """Dynamically generate label list"""
        return [sample["label"] for sample in self.samples]


class PPIDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        # 使用静态方法作为collate_fn
        kwargs['collate_fn'] = PPIDataLoader.collate_batch
        super(PPIDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    @staticmethod
    def collate_batch(batch):
        """静态方法，不依赖于实例状态"""
        # 提取和准备所需数据
        atom_graphs = [item['atom_graph'] for item in batch]
        residue_graphs = [item['residue_graph'] for item in batch]
        a2r_maps = [item['a2r_map'] for item in batch]
        labels = [item['label'] for item in batch]

        # 合并图
        atom_batch = Batch.from_data_list(atom_graphs)
        residue_batch = Batch.from_data_list(residue_graphs)

        # 创建稀疏张量邻接矩阵
        atom_adj_t = SparseTensor(
            row=atom_batch.edge_index[0],
            col=atom_batch.edge_index[1],
            value=atom_batch.edge_attr,
            sparse_sizes=(atom_batch.x.size(0), atom_batch.x.size(0))
        ).t()

        residue_adj_t = SparseTensor(
            row=residue_batch.edge_index[0],
            col=residue_batch.edge_index[1],
            value=residue_batch.edge_attr,
            sparse_sizes=(residue_batch.x.size(0), residue_batch.x.size(0))
        ).t()

        # 处理a2r_map - 需要根据batch偏移调整
        # 这部分需要特别注意，因为合并后节点索引会改变
        offset_maps = []
        atom_offset = 0
        residue_offset = 0

        for i, (a_graph, r_graph, a2r) in enumerate(zip(atom_graphs, residue_graphs, a2r_maps)):
            if i > 0:
                atom_offset += atom_graphs[i - 1].x.size(0)
                residue_offset += residue_graphs[i - 1].x.size(0)

            # 调整映射关系以考虑batch中的偏移
            adjusted_map = a2r + residue_offset
            offset_maps.append(adjusted_map)

        combined_a2r_map = torch.cat(offset_maps)

        # 为模型准备批次
        custom_batch = {
            'atom_x': atom_batch.x,
            'atom_adj_t': atom_adj_t,
            'residue_x': residue_batch.x,
            'residue_adj_t': residue_adj_t,
            'a2r_map': combined_a2r_map,
            'y': torch.cat(labels),
            'batch_size': len(batch),
            'train_mask': torch.ones(len(batch), dtype=torch.bool)
        }

        return custom_batch


if __name__ == '__main__':
    import pickle


    def load_data(pkl_path):
        with open(pkl_path, 'rb') as f:
            raw_data = pickle.load(f)
        print(f"Loaded data from {pkl_path}")
        return raw_data


    data = load_data(pkl_path='/gz-data/train355-r5.5-a2.3.pkl')
    protein_data = ProteinData(data)
    loader = PPIDataLoader(protein_data, batch_size=2, shuffle=True)
    all_target = []
    for batch in loader:
        # print(batch.keys())
        # print(f"Batch size: {batch['batch_size']}")
        # print(f"Atom features shape: {batch['atom_x'].shape}")
        # print(f"Residue features shape: {batch['residue_x'].shape}")
        # print(protein_data[3]['residue_graph'].x.size(1))
        all_target.append(batch['y'])
    all_target = torch.cat(all_target)
    num_pos = (all_target == 1).sum().item()
    num_neg = (all_target == 0).sum().item()
    total = num_pos + num_neg
    print("pos / neg:", num_neg / num_pos)