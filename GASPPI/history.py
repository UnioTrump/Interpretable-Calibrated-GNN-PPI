from typing import Optional

import torch
from torch import Tensor


class History(torch.nn.Module):
    r"""A historical embedding storage module."""
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        pin_memory = device is None or str(device) == 'cpu'
        self.emb = torch.empty(num_embeddings, embedding_dim, device=device,
                               pin_memory=pin_memory)       # 未初始化的嵌入矩阵，用于存储embeddings，pin_memory：是否固定存储GPU加速

        self._device = torch.device('cpu')

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.fill_(0)       # 初始化嵌入矩阵

    def _apply(self, fn):
        # Set the `_device` of the module without transfering `self.emb`.
        """控制 PyTorch 模块在设备间迁移时的行为，更新模块的设备信息，但避免移动 self.emb 张量"""
        self._device = fn(torch.zeros(1)).device
        return self

    @torch.no_grad()
    def pull(self, n_id: Optional[Tensor] = None) -> Tensor:
        """从嵌入矩阵中提取特定索引对应的嵌入向量"""
        out = self.emb
        if n_id is not None:
            assert n_id.device == self.emb.device
            out = out.index_select(0, n_id)
        return out.to(device=self._device)

    @torch.no_grad()
    def push(self, x, n_id: Optional[Tensor] = None,
             offset: Optional[Tensor] = None, count: Optional[Tensor] = None):

        if n_id is None and x.size(0) != self.num_embeddings:
            raise ValueError

        elif n_id is None and x.size(0) == self.num_embeddings:
            self.emb.copy_(x)       # 将x的值复制到self.emb中

        elif offset is None or count is None:
            assert n_id.device == self.emb.device
            self.emb[n_id] = x.to(self.emb.device)

        else:  # Push in chunks:
            src_o = 0       # 初始化源数据起始位置
            x = x.to(self.emb.device)
            for dst_o, c, in zip(offset.tolist(), count.tolist()):      # 遍历每个块
                self.emb[dst_o:dst_o + c] = x[src_o:src_o + c]          # 复制块数据
                src_o += c      # 更新区块指针

    def forward(self, *args, **kwargs):
        """"""
        raise NotImplementedError

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_embeddings}, '
                f'{self.embedding_dim}, emb_device={self.emb.device}, '
                f'device={self._device})')
