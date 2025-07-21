import torch
from torch_geometric.data import Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE


def add_gaussian_edge_weights(data: Data, sigma: float = 1.0) -> Data:
    """Computes and adds Gaussian kernel edge weights."""
    node_features = data.x
    edge_index = data.edge_index

    # Get the features of the source and target nodes for each edge
    row, col = edge_index
    feat_i = node_features[row]
    feat_j = node_features[col]

    # Compute the squared Euclidean distance
    dist = torch.sum((feat_i - feat_j) ** 2, dim=-1)

    # Apply the Gaussian kernel
    weights = torch.exp(-dist / (2 * sigma ** 2))

    data.edge_attr = weights.unsqueeze(1)  # Ensure shape is [num_edges, 1]
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
