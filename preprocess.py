import os
import torch
import pickle
from tqdm import tqdm
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from data_utils.data_utils import compute_fourier_features
from data_utils.utils import add_gaussian_edge_weights
from torch_geometric.transforms import AddLaplacianEigenvectorPE as LapPE
import config


def process_single_protein(sample, enable_fourier=True, enable_pe=True):
    """处理单个蛋白质样本
    
    Args:
        sample: 原始样本数据
        enable_fourier: 是否计算傅里叶特征
        enable_pe: 是否计算位置编码
    """
    # 转换为tensor
    atom_x = torch.FloatTensor(sample['a_node'])
    residue_x = torch.FloatTensor(sample['r_node'])
    a_edge_index = torch.LongTensor(sample['a_edge_index'])
    r_edge_index = torch.LongTensor(sample['r_edge_index'])
    a2r_map = torch.tensor(sample['a2r_map'])
    
    # 处理标签
    labels = sample['label']
    label_list = [int(char) for char in labels]
    y = torch.LongTensor(label_list)
    
    # 计算高斯边权重
    a_weights = add_gaussian_edge_weights(
        Data(x=atom_x, edge_index=a_edge_index),
        sigma=config.GAUSSIAN_SIGMA
    )
    r_weights = add_gaussian_edge_weights(
        Data(x=residue_x, edge_index=r_edge_index),
        sigma=config.GAUSSIAN_SIGMA
    )
    
    # 构建稀疏张量
    atom_adj_t = SparseTensor(
        row=a_weights.edge_index[0],
        col=a_weights.edge_index[1],
        value=a_weights.edge_attr,
        sparse_sizes=(len(a_weights.x), len(a_weights.x))
    ).t()
    
    residue_adj_t = SparseTensor(
        row=r_weights.edge_index[0],
        col=r_weights.edge_index[1],
        value=r_weights.edge_attr,
        sparse_sizes=(len(r_weights.x), len(r_weights.x))
    ).t()
    
    processed_data = {
        'atom_x': atom_x,
        'residue_x': residue_x,
        'atom_adj_t': atom_adj_t,
        'residue_adj_t': residue_adj_t,
        'a_edge_index': a_edge_index,
        'r_edge_index': r_edge_index,
        'a2r_map': a2r_map,
        'y': y
    }
    
    if enable_pe:
        # 计算位置编码
        graph_transformer = LapPE(k=16, attr_name='r_pe')
        r_pe_data = Data(x=residue_x, edge_index=r_edge_index)
        r_pe_data = graph_transformer(r_pe_data)
        processed_data['r_pe'] = r_pe_data.r_pe
    
    if enable_fourier:
        # 计算傅里叶特征
        r_fourier = compute_fourier_features(
            residue_x,
            r_edge_index,
            threshold=config.FOURIER_THRESHOLD
        )
        processed_data['r_fourier'] = r_fourier
    
    return processed_data

def preprocess_dataset(data_path, save_dir, enable_fourier=True, enable_pe=True):
    """预处理整个数据集
    
    Args:
        data_path: 原始数据路径
        save_dir: 预处理结果保存路径
        enable_fourier: 是否计算傅里叶特征
        enable_pe: 是否计算位置编码
    """
    print(f"开始处理数据集: {data_path}")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载原始数据
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # 处理每个样本
    processed_data = []
    for idx, sample in enumerate(tqdm(data, desc="预处理蛋白质")):
        try:
            processed_sample = process_single_protein(
                sample, 
                enable_fourier=enable_fourier,
                enable_pe=enable_pe
            )
            processed_data.append(processed_sample)
                
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {str(e)}")
            continue
    
    # 保存剩余样本
    if processed_data:
        save_path = os.path.join(save_dir, f'batch_final.pkl')
        torch.save(processed_data, save_path)
    
    print(f"预处理完成，结果保存在: {save_dir}")

if __name__ == "__main__":
    # 设置预处理参数
    ENABLE_FOURIER = True
    ENABLE_PE = True
    
    # 预处理训练集
    preprocess_dataset(
        data_path=config.DATA_PATH,
        save_dir='/../gz-data/Pretrain/',
        enable_fourier=ENABLE_FOURIER,
        enable_pe=ENABLE_PE
    )
    
    # 预处理验证集
    preprocess_dataset(
        data_path=config.VAL_DATA_PATH,
        save_dir='/../gz-data/Val',
        enable_fourier=ENABLE_FOURIER,
        enable_pe=ENABLE_PE
    )
    
    # 预处理测试集（如果有的话）
    if hasattr(config, 'TEST_DATA_PATH'):
        preprocess_dataset(
            data_path=config.TEST_DATA_PATH,
            save_dir='processed_data/test',
            enable_fourier=ENABLE_FOURIER,
            enable_pe=ENABLE_PE
        )

