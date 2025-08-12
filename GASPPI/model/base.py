import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import ModuleList, Linear
from torch_geometric.nn import JumpingKnowledge

class BaseEncoder(nn.Module):
    """可扩展的GNN编码器基础框架"""
    def __init__(self, in_channels: int, hidden_dims: list, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = ModuleList()
        self.norms = ModuleList()
        self.heads = heads # heads 属性在子类中需要

        if in_channels != hidden_dims[0]:
            self.in_proj = Linear(in_channels, hidden_dims[0])
        else:
            self.in_proj = None
            
        self.layer_dims = [hidden_dims[0]] + hidden_dims
        self._build_layers()

        self.jk = JumpingKnowledge(mode='cat')
        self.out_dim = sum(hidden_dims)
        
    def _build_layers(self):
        """用于子类重载的层构建方法"""
        raise NotImplementedError("Subclasses must implement the _build_layers method")

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.in_proj:
            x = self.in_proj(x)

        xs = [x]
        for conv, norm in zip(self.convs, self.norms):
            # 将额外的参数传递给卷积层
            x = conv(x, *args, **kwargs)
            x = F.relu(x)
            x = norm(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        
        return self.jk(xs[1:]) 