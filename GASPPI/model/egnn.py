import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from e3nn.o3 import Irreps
from torch_geometric.nn.models import Equiformer

class EquiformerEncoder(nn.Module):
    """
    使用Equiformer的等变图编码器
    """
    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 out_channels, 
                 n_layers=4,
                 num_heads=8,
                 edge_dim=None):
        super().__init__()
        
        # 定义输入不变量和等变量的表示
        irreps_in = Irreps(f"{in_channels}x0e")  # 标量特征
        irreps_hidden = Irreps(f"{hidden_channels}x0e + {hidden_channels}x1o")  # 标量+向量特征
        irreps_out = Irreps(f"{out_channels}x0e")  # 输出标量特征
        
        # 使用Equiformer
        self.model = Equiformer(
            irreps_in=irreps_in,
            irreps_hidden=irreps_hidden,
            irreps_out=irreps_out,
            irreps_edge_attr=Irreps(f"{edge_dim}x0e") if edge_dim else None,
            num_layers=n_layers,
            num_heads=num_heads,
            hidden_channels=hidden_channels,
            reduction='sum',  # 或者使用 'mean'
            use_xyz_features=True,  # 使用空间坐标作为额外特征
            drop_path_rate=0.1
        )
        
    def forward(self, x, pos, edge_index, edge_attr=None, batch=None):
        """
        参数:
            x: 节点特征 [N, in_channels]
            pos: 节点的3D坐标 [N, 3]
            edge_index: 边的连接关系 [2, E]
            edge_attr: 边特征 [E, edge_dim]
            batch: 批处理指示器 [N]
        返回:
            node_features: 更新后的节点特征 [N, out_channels]
        """
        # 构建数据字典
        data_dict = {
            'x': x,
            'pos': pos,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'batch': batch
        }
        
        # 通过Equiformer处理
        out = self.model(data_dict)
        
        return out
