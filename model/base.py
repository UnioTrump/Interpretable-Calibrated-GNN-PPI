import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import math
from torch_sparse import SparseTensor
import config
device = config.DEVICE

class PPIConv(nn.Module):
    def __init__(self, in_channels: int, hid_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = hid_channels

        self.weight = nn.Parameter(torch.FloatTensor(in_channels, hid_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_channels)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, H0, alpha, lamda, l):
        beta = min(1.0, math.log(lamda / l + 1))
        Hl = torch.spmm(adj, x)

        r = (1 - alpha) * Hl + alpha * H0
        r = (1 - beta) * r + beta * torch.mm(r, self.weight)
        return r


class GNNEncoder(nn.Module):
    def __init__(self, in_channels: int, hid_dim: int, alpha, lamda, training, dropout: float):
        super().__init__()

        self.convs = nn.ModuleList()
        self.lin = nn.ModuleList()
        self.act = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.training = training
        for _ in range(config.NUM_LAYER):
            phi = PPIConv(in_channels=hid_dim, hid_channels=hid_dim)
            self.convs.append(phi)
        self.lin = nn.Linear(in_channels, hid_dim)

        self.out_dim: int = hid_dim

    def forward(self, x: Tensor, adj: SparseTensor):
        _layer = []
        x = F.dropout(x, self.dropout, training=self.training)
        _layer.append(self.act(self.lin(x)))
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.act(conv(x, adj, _layer[0], self.alpha, self.lamda, i + 1))
        x = F.dropout(x, self.dropout, training=self.training)

        return x


class PPI(nn.Module):
    """
    Args:
        in_channels (int): input channels
        hid_dim (int): hidden channels
        dropout (float): dropout rate
        lamda (float): cal weight of W(l)
        alpha (float): weight of W(l)
    """
    def __init__(self, in_channels: int, hid_dim: int,
                 dropout: float, lamda: float, alpha: float, training=True):
        super().__init__()

        self.GCN = GNNEncoder(in_channels=512, hid_dim=hid_dim,
                              alpha=alpha, lamda=lamda, dropout=dropout, training=training)
        self.trf = nn.TransformerEncoderLayer(d_model=606, nhead=6)
        self.classifier = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hid_dim // 2, 1)
        )
        self.reduction = nn.Linear(2782, 512)

    def forward(self, ax: Tensor, bx: Tensor, cx: Tensor, adj: SparseTensor):
        """
        ax: AAINDEX and BLOSUM62 martix:
            shape:[N, 566+20]
        bx: Obtained from ProtTrans:
            shape:[N, 1280]
        """
        ax.to(device)
        bx.to(device)
        cx.to(device)
        ax = ax.float()
        bx = bx.float()

        w = self.trf(ax)
        x = torch.cat([w, bx, cx], dim=1)   # [n_sample, 2762]
        x = self.reduction(x)
        out = self.GCN(x, adj)

        return self.classifier(out)
