from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, Dropout, ReLU
from torch_sparse import SparseTensor
from torch_geometric.nn import GATConv

from .base import ScalableGNN

class PPI(ScalableGNN):
    def __init__(self, num_nodes: int, in_channels, hidden_channels: int,
                 hidden_heads: int, out_channels: int, out_heads: int,
                 num_layers: int, dropout: float = 0.0,
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None):
        super().__init__(num_nodes, hidden_channels * hidden_heads, num_layers,
                         pool_size, buffer_size, device)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.hidden_heads = hidden_heads
        self.out_channels = out_channels
        self.out_heads = out_heads
        self.dropout = dropout
        self.att = Linear(out_channels * out_heads, out_channels * out_heads)
        self.v = Linear(out_channels * out_heads, 1)

        # 定义多层神经网络
        self.convs = ModuleList()

        # 第一层Linear
        conv = Sequential(
            Linear(in_channels, hidden_channels * hidden_heads),
            ReLU(),
            Dropout(dropout)
        )
        self.convs.append(conv)

        for i in range(num_layers - 1):
            in_dim = hidden_channels * hidden_heads
            conv = GATConv(in_dim, hidden_channels, hidden_heads, concat=True,
                           dropout=dropout, add_self_loops=False)
            self.convs.append(conv)

        # 输出层,，不是分类头
        conv = GATConv(hidden_channels * hidden_heads, out_channels, out_heads,
                       concat=False, dropout=dropout, add_self_loops=False)
        self.convs.append(conv)

        self.reg_modules = self.convs  # 正则化模块：指权重需要应用权重衰减（weight decay）的层，如卷积层、全连接层
        self.nonreg_modules = ModuleList()  # 非正则化模块：不需要应用权重衰减的层，如：BatchNorm 层，偏置项（bias），自定义参数（如缩放因子）

    def reset_parameters(self):
        super().reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, adj_t: SparseTensor, *args) -> Tensor:
        # 第一层是Sequential(Linear, ReLU, Dropout)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[0](x)  # 直接调用Sequential
        x = F.elu(x)
        x = self.push_and_pull(self.histories[0], x, *args)

        # 中间的GATConv层
        for conv, history in zip(self.convs[1:-1], self.histories[1:]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv((x, x[:adj_t.size(0)]), adj_t)
            x = F.elu(x)  # ELU激活函数使激活的平均值接近零。均值激活接近于零可以使学习更快，因为它们使梯度更接近自然梯度。
            x = self.push_and_pull(history, x, *args)

        # 最后一层GATConv
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1]((x, x[:adj_t.size(0)]), adj_t)

        att = self.v(torch.tanh(self.att(x)))
        att_score = F.softmax(att, dim=1)
        scored_out = x * att_score

        return scored_out

    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, state):
        raise NotImplementedError
