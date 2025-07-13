from typing import Optional, Callable

import torch
from torch import Tensor
from torch.cuda import Stream

# 删除对不存在的操作的引用，添加简单的Python实现
# synchronize = torch.ops.torch_geometric_autoscale.synchronize
# read_async = torch.ops.torch_geometric_autoscale.read_async
# write_async = torch.ops.torch_geometric_autoscale.write_async

# 简单的Python替代实现
def synchronize():
    # 简单的同步操作，对PyTorch流进行同步
    torch.cuda.synchronize()

def read_async(src, offset, count, index, cuda_buffer, cpu_buffer):
    # 简单的异步读取实现
    if offset is not None and count is not None:
        # 从特定偏移和计数读取
        if index is None:
            cuda_buffer.copy_(src[offset:offset+count])
        else:
            cuda_buffer.copy_(src[index])
    else:
        # 使用索引读取
        if index is not None:
            cuda_buffer.copy_(src.index_select(0, index))

def write_async(src, offset, count, dst):
    # 简单的异步写入实现
    if offset is not None and count is not None:
        # 写入到特定偏移和计数
        dst[offset:offset+count].copy_(src)
    else:
        # 无偏移写入
        dst.copy_(src)


class AsyncIOPool(torch.nn.Module):
    """
    异步写入是将 GPU 上计算好的节点嵌入，通过非阻塞方式、写入到大图的嵌入表（通常在 CPU 上）的过程，它优化了训练效率和内存使用，是处理大图训练时不可或缺的技术。
    其可以在训练当前batch的同时，将上一个batch的结果写入cpu
    """
    def __init__(self, pool_size: int, buffer_size: int, embedding_dim: int):
        super().__init__()

        self.pool_size = pool_size
        self.buffer_size = buffer_size
        self.embedding_dim = embedding_dim

        self._device = torch.device('cpu')
        self._pull_queue = []
        self._push_cache = [None] * pool_size
        self._push_streams = [None] * pool_size
        self._pull_streams = [None] * pool_size
        self._cpu_buffers = [None] * pool_size
        self._cuda_buffers = [None] * pool_size
        self._pull_index = -1
        self._push_index = -1

    def _apply(self, fn: Callable) -> None:
        self._device = fn(torch.zeros(1)).device
        return self

    def _pull_stream(self, idx: int) -> Stream:
        if self._pull_streams[idx] is None:
            assert str(self._device)[:4] == 'cuda'
            self._pull_streams[idx] = torch.cuda.Stream(self._device)
        return self._pull_streams[idx]

    def _push_stream(self, idx: int) -> Stream:
        if self._push_streams[idx] is None:
            assert str(self._device)[:4] == 'cuda'
            self._push_streams[idx] = torch.cuda.Stream(self._device)
        return self._push_streams[idx]

    def _cpu_buffer(self, idx: int) -> Tensor:
        if self._cpu_buffers[idx] is None:
            self._cpu_buffers[idx] = torch.empty(self.buffer_size,
                                                 self.embedding_dim,
                                                 pin_memory=True)
        return self._cpu_buffers[idx]

    def _cuda_buffer(self, idx: int) -> Tensor:
        if self._cuda_buffers[idx] is None:
            assert str(self._device)[:4] == 'cuda'
            self._cuda_buffers[idx] = torch.empty(self.buffer_size,
                                                  self.embedding_dim,
                                                  device=self._device)
        return self._cuda_buffers[idx]

    @torch.no_grad()
    def async_pull(self, src: Tensor, offset: Optional[Tensor],
                   count: Optional[Tensor], index: Tensor) -> None:
        # Start pulling `src` at ([offset, count] and index positions:
        self._pull_index = (self._pull_index + 1) % self.pool_size
        data = (self._pull_index, src, offset, count, index)
        self._pull_queue.append(data)
        if len(self._pull_queue) <= self.pool_size:
            self._async_pull(self._pull_index, src, offset, count, index)

    @torch.no_grad()
    def _async_pull(self, idx: int, src: Tensor, offset: Optional[Tensor],
                    count: Optional[Tensor], index: Tensor) -> None:
        with torch.cuda.stream(self._pull_stream(idx)):
            read_async(src, offset, count, index, self._cuda_buffer(idx),
                       self._cpu_buffer(idx))

    @torch.no_grad()
    def synchronize_pull(self) -> Tensor:
        # Synchronize the next pull command:
        idx = self._pull_queue[0][0]
        synchronize()
        torch.cuda.synchronize(self._pull_stream(idx))
        return self._cuda_buffer(idx)

    @torch.no_grad()
    def free_pull(self) -> None:
        # Free the buffer space and start pulling from remaining queue:
        self._pull_queue.pop(0)
        if len(self._pull_queue) >= self.pool_size:
            data = self._pull_queue[self.pool_size - 1]
            idx, src, offset, count, index = data
            self._async_pull(idx, src, offset, count, index)
        elif len(self._pull_queue) == 0:
            self._pull_index = -1

    @torch.no_grad()
    def async_push(self, src: Tensor, offset: Tensor, count: Tensor,
                   dst: Tensor) -> None:
        # Start pushing `src` to ([offset, count] and index positions to `dst`:
        self._push_index = (self._push_index + 1) % self.pool_size       # 循环使用缓冲区
        self.synchronize_push(self._push_index)     # 确保前一个操作完成
        self._push_cache[self._push_index] = src    # 防止被覆盖
        with torch.cuda.stream(self._push_stream(self._push_index)):        # 在专用流中异步执行写入
            write_async(src, offset, count, dst)

    @torch.no_grad()
    def synchronize_push(self, idx: Optional[int] = None) -> None:
        # Synchronize the push command of stream `idx` or all commands:
        if idx is None:
            for idx in range(self.pool_size):
                self.synchronize_push(idx)
            self._push_index = -1
        else:
            torch.cuda.synchronize(self._push_stream(idx))
            self._push_cache[idx] = None

    def forward(self, *args, **kwargs):
        """"""
        raise NotImplementedError

    def __repr__(self):
        return (f'{self.__class__.__name__}(pool_size={self.pool_size}, '
                f'buffer_size={self.buffer_size}, '
                f'embedding_dim={self.embedding_dim}, '
                f'device={self._device})')
