import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import AntiSymmetricConv, TransformerConv
from torch import Tensor
from torch_sparse import SparseTensor
import config

device = config.DEVICE


class PPI(nn.Module):
    """
        Args:
            hid_dim (int): Hidden dimension size.
            heads (int, optional): Number of attention heads.
            dropout (float, optional): Dropout probability.
    """

    def __init__(self, hid_dim: int, heads: int, dropout: float, bi: bool):
        super().__init__()
        self.dropout = dropout
        self.hid_dim = hid_dim
        self.out_dim: int = hid_dim
        self.bi = bi

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.trf = nn.GRU(input_size=567, hidden_size=config.gru_hid_dim, num_layers=1, bidirectional=bi)
        input_size=config.gru_hid_dim*2+1024+1152
        self.gru = nn.GRU(input_size=input_size, hidden_size=hid_dim, num_layers=1, bidirectional=bi)  # 512+1024+1152=2688
        if bi:
            for _ in range(config.NUM_LAYER):
                phi = TransformerConv(in_channels=hid_dim*2, out_channels=hid_dim*2, heads=heads, concat=False, beta=False,
                                      dropout=dropout, edge_dim=1)
                conv1 = AntiSymmetricConv(in_channels=hid_dim*2, phi=phi, act='ReLU')
                self.convs.append(conv1)
                self.norms.append(nn.LayerNorm(self.hid_dim*2))
            self.classifier = nn.Sequential(
                nn.Linear(hid_dim*2, hid_dim // 2),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hid_dim // 2, 1)
            )
        else:
            for _ in range(config.NUM_LAYER):
                phi = TransformerConv(in_channels=hid_dim, out_channels=hid_dim, heads=heads, concat=False, beta=False,
                                      dropout=dropout, edge_dim=1)
                conv1 = AntiSymmetricConv(in_channels=hid_dim, phi=phi, act='ReLU')
                self.convs.append(conv1)
                self.norms.append(nn.LayerNorm(self.hid_dim))

            self.classifier = nn.Sequential(
                nn.Linear(hid_dim, hid_dim // 2),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hid_dim // 2, 1)
            )

    def forward(self, ax: Tensor, bx: Tensor, cx: Tensor, adj: SparseTensor):
        """
        Parameters:

            ax: Tensor
                The AAINDEX features.
            bx: Tensor
                The ESM-C 600m features.
            cx: Tensor
                The ProtTrans features.
            adj: SparseTensor
                The adjacency matrix in COO format, representing the graph structure.
        """
        ax.to(device)
        bx.to(device)
        cx.to(device)
        row, col, edge_attr = adj.coo()
        edge_attr = edge_attr.view(-1, 1)
        edge_index = torch.stack([row, col]).long()

        w, _ = self.trf(ax)
        x = torch.cat([w, bx, cx], dim=1)
        x, _ = self.gru(x)


        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = norm(x)
            x = F.dropout(x, p=self.dropout, training=True)

        return self.classifier(x)
