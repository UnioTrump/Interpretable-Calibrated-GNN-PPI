import torch
import torch.nn.functional as F

class CosineSimilarity:
    @staticmethod
    def cosine_similarity_attention(embeddings):
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        sm = torch.matmul(embeddings_norm, embeddings_norm.T)
        return sm
    
    @staticmethod
    def create_sparse_attention_tensor(dense_attention, threshold=0.5):
        # 创建下三角掩码（不包括对角线）
        L = dense_attention.shape[0]
        lower_triangular_mask = torch.tril(torch.ones(L, L, dtype=torch.bool), diagonal=-1)
        
        # 应用阈值和下三角掩码
        threshold_mask = dense_attention > threshold
        combined_mask = lower_triangular_mask & threshold_mask
        
        # 获取非零元素的位置和值
        row, col = torch.where(combined_mask)
        values = dense_attention[row, col]
        
        # 创建edge_index和edge_attr
        edge_index = torch.stack([row, col])
        edge_attr = values
        
        # 创建稀疏张量
        sparse_tensor = torch.sparse_coo_tensor(
            indices=edge_index,
            values=edge_attr,
            size=dense_attention.shape
        )
        
        return sparse_tensor, edge_index, edge_attr
    
    @staticmethod
    def remove_diagonal_from_sparse(sparse_tensor, edge_index, edge_attr):
        # 获取当前稀疏张量的索引和值
        indices = sparse_tensor.coalesce().indices()
        values = sparse_tensor.coalesce().values()
        
        # 创建非对角线掩码
        non_diag_mask = indices[0] != indices[1]
        
        # 过滤非对角线元素
        new_indices = indices[:, non_diag_mask]
        new_values = values[non_diag_mask]
        
        # 创建新的稀疏张量
        new_sparse_tensor = torch.sparse_coo_tensor(
            indices=new_indices,
            values=new_values,
            size=sparse_tensor.shape
        )
        
        # 同时更新edge_index和edge_attr
        new_edge_index = edge_index[:, non_diag_mask]
        new_edge_attr = edge_attr[non_diag_mask]
        
        return new_sparse_tensor, new_edge_index, new_edge_attr
    
    @classmethod
    def compute_attention(cls, embeddings, temperature=0.5, threshold=0.5, remove_diagonal=True):
        # 计算稠密注意力矩阵
        dense_attention = cls.cosine_similarity_attention(embeddings)
        
        # 转换为稀疏张量，同时获取edge_index和edge_attr
        sparse_attention, edge_index, edge_attr = cls.create_sparse_attention_tensor(
            dense_attention, threshold
        )
        
        # 如果需要，移除对角线
        if remove_diagonal:
            sparse_attention, edge_index, edge_attr = cls.remove_diagonal_from_sparse(
                sparse_attention, edge_index, edge_attr
            )
        
        return sparse_attention, edge_index, edge_attr

# 使用示例
if __name__ == "__main__":
    import pickle as p
    
    # 加载数据
    with open('/../../gz-data/Pretrain/esmc_Test_60.pkl', 'rb') as f:
        d = p.load(f)
    embeddings = d[33]['x']
    
    # 使用模块计算注意力
    sparse_attn, edge_index, edge_attr = CosineSimilarity.compute_attention(
        embeddings, 
        threshold=0.7
    )
    
    print("稀疏注意力张量:")
    print(sparse_attn)
    print(f"稀疏注意力张量非零元素数量: {sparse_attn._nnz()}")
    
    # 输出edge_index信息
    print(f"edge_index形状: {edge_index.shape}")
    print(f"edge_attr形状: {edge_attr.shape}")