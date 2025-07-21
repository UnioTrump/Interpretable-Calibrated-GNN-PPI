import torch
from torch_geometric.data import Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE


def add_gaussian_edge_weights(data: Data, sigma: float = 1.0) -> Data:
    """Computes and adds Gaussian kernel edge weights."""
    node_features = data.x
    edge_index = data.edge_index

    row, col = edge_index
    feat_i = node_features[row]
    feat_j = node_features[col]

    d = torch.sum((feat_i - feat_j) ** 2, dim=-1)
    weights = torch.exp(-d / (2 * sigma ** 2))
    data.edge_attr = weights.unsqueeze(1)
    return data


def add_laplacian_pe(data: Data, pe_dim: int) -> Data:
    """Computes and adds Laplacian Positional Encodings."""
    transform = AddLaplacianEigenvectorPE(
        k=pe_dim,
        attr_name='lap_pe',
        is_undirected=True
    )
    data = transform(data)
    return data
