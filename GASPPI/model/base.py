from typing import Optional, Callable, Dict, Any

import warnings

import torch
from torch import Tensor
from torch_sparse import SparseTensor

# 修改导入方式，使用相对导入避免循环引用
from ..history import History
from ..pool import AsyncIOPool


class ScalableGNN(torch.nn.Module):
    r"""An abstract class for implementing scalable GNNs via historical
    embeddings.
    This class will take care of initializing :obj:`num_layers - 1` historical
    embeddings, and provides a convenient interface to push recent node
    embeddings to the history, and to pull previous embeddings from the
    history.
    In case historical embeddings are stored on the CPU, they will reside
    inside pinned memory, which allows for asynchronous memory transfers of
    historical embeddings.
    For this, this class maintains a :class:`AsyncIOPool` object that
    implements the underlying mechanisms of asynchronous memory transfers as
    described in our paper.

    Args:
        num_nodes (int): The number of nodes in the graph.
        hidden_channels (int): The number of hidden channels of the model.
            As a current restriction, all intermediate node embeddings need to
            utilize the same number of features.
        num_layers (int): The number of layers of the model.
        pool_size (int, optional): The number of pinned CPU buffers for pulling
            histories and transfering them to GPU.
            Needs to be set in order to make use of asynchronous memory
            transfers. (default: :obj:`None`)
        buffer_size (int, optional): The size of pinned CPU buffers, i.e. the
            maximum number of out-of-mini-batch nodes pulled at once.
            Needs to be set in order to make use of asynchronous memory
            transfers. (default: :obj:`None`)
    """
    '''
        num_nodes（整数）： 图中的节点数。
        hidden_channels （int）：隐藏通道数： 模型的隐藏通道数。
        num_layers (int)： 模型的层数： 模型的层数。
        pool_size （int，可选）： 用于提取历史记录并将其传输到 GPU 的 CPU 缓冲区的数量。需要设置此值才能使用异步内存传输。(默认值: :obj:`None`)
        buffer_size（int，可选项）： 固定 CPU 缓冲区的大小，即一次提取的最大迷你批次外节点数。需要设置该值才能使用异步内存传输。(默认值： :obj:`None`)
    '''
    def __init__(self, num_nodes: int, hidden_channels: int, num_layers: int,
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.pool_size = num_layers - 1 if pool_size is None else pool_size
        self.buffer_size = buffer_size

        self.histories = torch.nn.ModuleList([
            History(num_nodes, hidden_channels, device)
            for _ in range(num_layers - 1)
        ])

        self.pool: Optional[AsyncIOPool] = None
        self._async = False
        self.__out: Optional[Tensor] = None

    @property
    def emb_device(self):
        return self.histories[0].emb.device

    @property
    def device(self):
        return self.histories[0]._device

    def _apply(self, fn: Callable) -> None:
        super()._apply(fn)
        # We only initialize the AsyncIOPool in case histories are on CPU:
        if (str(self.emb_device) == 'cpu' and str(self.device)[:4] == 'cuda'
                and self.pool_size is not None
                and self.buffer_size is not None):
            self.pool = AsyncIOPool(self.pool_size, self.buffer_size,
                                    self.histories[0].embedding_dim)
            self.pool.to(self.device)
        return self

    def reset_parameters(self):
        for history in self.histories:
            history.reset_parameters()

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
        r"""Enhances the call of forward propagation by immediately start
        pulling historical embeddings for all layers asynchronously.
        After forward propogation is completed, the push of node embeddings to
        the histories will be synchronized.

        For example, given a mini-batch with node indices
        :obj:`n_id = [0, 1, 5, 6, 7, 3, 4]`, where the first 5 nodes
        represent the mini-batched nodes, and nodes :obj:`3` and :obj:`4`
        denote out-of-mini-batched nodes (i.e. the 1-hop neighbors of the
        mini-batch that are not included in the current mini-batch), then
        other input arguments should be given as:

        .. code-block:: python

            batch_size = 5
            offset = [0, 5]
            count = [2, 3]

        Args:
            x (Tensor, optional): Node feature matrix. (default: :obj:`None`)
            adj_t (SparseTensor, optional) The sparse adjacency matrix.
                (default: :obj:`None`)
            batch_size (int, optional): The in-mini-batch size of nodes.
                (default: :obj:`None`)
            n_id (Tensor, optional): The global indices of mini-batched and
                out-of-mini-batched nodes. (default: :obj:`None`)
            offset (Tensor, optional): The offset of mini-batched nodes inside
                a utilize a contiguous memory layout. (default: :obj:`None`)
            count (Tensor, optional): The number of mini-batched nodes inside a
                contiguous memory layout. (default: :obj:`None`)
            loader (EvalSubgraphLoader, optional): A subgraph loader used for
                evaluating the given GNN in a layer-wise fashsion.
        """
        '''
            x （Tensor，可选）：节点特征矩阵。（默认： ：obj：'None'）
            adj_t （SparseTensor，可选） 稀疏邻接矩阵。（默认： ：obj：'None'）
            batch_size （int， optional）：节点的 in-mini-batch 大小。（默认： ：obj：'None'）
            n_id （Tensor，可选）：小批量节点和非小批量节点的全局索引。（默认： ：obj：'None'）
            offset （Tensor，可选）：利用连续内存布局的内部小批量节点的偏移量。（默认： ：obj：'None'）
            count （Tensor，可选）：连续内存布局内的微型批处理节点数。（默认： ：obj：'None'）
            loader （EvalSubgraphLoader，可选）：一个子图加载器，用于在逐层布局中评估给定的 GNN。
        '''
        # 确保输入张量都在正确的设备上
        target_device = self.device
        if x is not None:
            x = x.to(target_device)
        if n_id is not None:
            n_id = n_id.to(target_device)
        if offset is not None:
            offset = offset.to(target_device)
        if count is not None:
            count = count.to(target_device)
        # SparseTensor 需要特殊处理，因为它不是标准的torch.Tensor
        # adj_t通常已经在GPU上了，但如果不是，我们需要确保它在正确的设备上
        # 注意：SparseTensor的to()方法可能与标准Tensor不同
        if adj_t is not None and hasattr(adj_t, 'device') and str(adj_t.device()) != str(target_device):
            print(f"Warning: adj_t is on {adj_t.device()}, moving to {target_device}")
            try:
                adj_t = adj_t.to(target_device)
            except:
                print("Failed to move adj_t, this may cause device mismatch errors")

        # We only perform asynchronous history transfer in case the following
        # conditions are met:
        self._async = (self.pool is not None and batch_size is not None
                       and n_id is not None and offset is not None
                       and count is not None)

        if (batch_size is not None and not self._async
                and str(self.emb_device) == 'cpu'
                and str(self.device)[:4] == 'cuda'):
            warnings.warn('Asynchronous I/O disabled, although history and '
                          'model sit on different devices.')

        if self._async:     # 如果满足条件则进行异步传输
            for hist in self.histories:
                self.pool.async_pull(hist.emb, None, None, n_id[batch_size:])

        out = self.forward(x, adj_t, batch_size, n_id, offset, count, **kwargs)

        if self._async:
            for hist in self.histories:
                self.pool.synchronize_push()

        self._async = False

        return out

    def push_and_pull(self, history, x: Tensor,
                      batch_size: Optional[int] = None,
                      n_id: Optional[Tensor] = None,
                      offset: Optional[Tensor] = None,
                      count: Optional[Tensor] = None) -> Tensor:
        r"""Pushes and pulls information from :obj:`x` to :obj:`history` and
        vice versa."""
        # print("begin push and pull function")     # make sure that fun can do correctly
        if n_id is None and x.size(0) != self.num_nodes:
            return x  # Do nothing...

        if n_id is None and x.size(0) == self.num_nodes:
            history.push(x)
            return x

        assert n_id is not None

        if batch_size is None:
            history.push(x, n_id)
            return x

        if not self._async:
            history.push(x[:batch_size], n_id[:batch_size], offset, count)
            h = history.pull(n_id[batch_size:])
            return torch.cat([x[:batch_size], h], dim=0)

        else:
            out = self.pool.synchronize_pull()[:n_id.numel() - batch_size]
            self.pool.async_push(x[:batch_size], offset, count, history.emb)
            out = torch.cat([x[:batch_size], out], dim=0)
            self.pool.free_pull()
            return out

    @property
    def _out(self):
        if self.__out is None:
            self.__out = torch.empty(self.num_nodes, self.out_channels,
                                     pin_memory=True)
        return self.__out

    @torch.no_grad()
    def forward_layer(self, layer: int, x: Tensor, adj_t: SparseTensor,
                      state: Dict[str, Any]) -> Tensor:
        raise NotImplementedError
