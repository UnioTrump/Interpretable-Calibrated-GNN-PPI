from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
import numpy as np
from torch_geometric.transforms import AddLaplacianEigenvectorPE


def add_gaussian_edge_weights(data: Data, sigma: float = 1.0) -> Data:
    """
    Computes edge weights based on the Gaussian kernel of node features and adds
    them to the data object as `edge_attr`.

    Args:
        data (Data): The graph data object, containing `data.x` and `data.edge_index`.
        sigma (float): The sigma parameter for the Gaussian kernel.

    Returns:
        Data: The data object with `edge_attr` added.
    """
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
    """
    Computes the Laplacian Positional Encodings for the graph and adds them
    to the data object.

    Args:
        data (Data): The graph data object.
        pe_dim (int): The dimension of the positional encodings.

    Returns:
        Data: The data object with `lap_pe` added.
    """
    # This transform computes the PE based on the graph's connectivity
    # and attaches it as 'data.lap_pe'
    transform = AddLaplacianEigenvectorPE(
        k=pe_dim,
        attr_name='lap_pe',
        is_undirected=True
    )
    data = transform(data)
    return data
