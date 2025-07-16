from typing import Optional, Callable, Dict, Any

import warnings

import torch
from torch import Tensor
from torch.nn import Module, Linear, LayerNorm, ELU, Sigmoid
from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor

# 修改导入方式，使用相对导入避免循环引用


class GatedGNNBlock(Module):
    """
    一个封装了GAT卷积、层归一化、激活和门控残差连接的构建块。
    这是构建深度、稳定图神经网络的核心模块。

    信息流:
    1. GAT卷积，与邻居节点进行信息交互。
    2. LayerNorm，稳定新生成的节点特征。
    3. ELU激活，进行非线性变换。
    4. 门控机制，动态融合原始特征和变换后的特征。
    """
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        # GAT卷积层 - 设置concat=False，使得多头输出被平均，保持维度不变
        self.conv = GATConv(in_channels, out_channels, heads=heads, concat=False, dropout=dropout, add_self_loops=False)
        # 层归一化 - 作用在out_channels上
        self.norm = LayerNorm(out_channels)
        # 激活函数
        self.act = ELU()
        # 门控机制的线性层
        # 现在x_in和x_transformed维度相同，都是out_channels
        self.gate_linear = Linear(2 * out_channels, out_channels)
        self.gate_act = Sigmoid()

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.norm.reset_parameters()
        self.gate_linear.reset_parameters()

    def forward(self, x: Tensor, adj_t: SparseTensor) -> Tensor:
        # 保存原始输入，用于残差/门控连接
        x_in = x

        # 1. 特征变换 F(x_in)
        # GAT卷积 -> LayerNorm -> ELU激活
        x_transformed = self.conv((x, x[:adj_t.size(0)]), adj_t)
        x_transformed = self.norm(x_transformed)
        x_transformed = self.act(x_transformed)

        # 2. 学习门控信号 z
        gate_input = torch.cat([x_in, x_transformed], dim=-1)
        z = self.gate_act(self.gate_linear(gate_input))

        # 3. 门控更新
        # 动态融合原始特征(x_in)和变换后的特征(x_transformed)
        x_out = (1 - z) * x_in + z * x_transformed

        return x_out


class ScalableGNN(torch.nn.Module):
    r"""An abstract class for implementing scalable GNNs."""
    def __init__(self, num_nodes: int, hidden_channels: int, num_layers: int):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

    def reset_parameters(self):
        pass

    def __call__(
        self,
        x: Optional[Tensor] = None,
        adj_t: Optional[SparseTensor] = None,
        batch_size: Optional[int] = None,
        n_id: Optional[Tensor] = None,
        offset: Optional[Tensor] = None,
        count: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        return self.forward(x, adj_t, batch_size, n_id, offset, count, **kwargs)

    @torch.no_grad()
    def forward_layer(self, layer: int, x: Tensor, adj_t: SparseTensor,
                      state: Dict[str, Any]) -> Tensor:
        raise NotImplementedError
