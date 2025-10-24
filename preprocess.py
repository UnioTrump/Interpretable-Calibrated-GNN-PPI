import os
import torch
import pickle
from tqdm import tqdm
from torch_geometric.data import Data
from data_utils.cos_sim import CosineSimilarity
from torch_geometric.transforms import AddLaplacianEigenvectorPE as LapPE
import argparse
import config

def frequency_filtering(eigenvalues, x_low, x_high):

    num_nodes = x_low.shape[0]
    eigenvalues_reshaped_i = eigenvalues.view(-1, 1)
    eigenvalues_reshaped_j = eigenvalues.view(1, -1)
    sum_matrix = eigenvalues_reshaped_i + eigenvalues_reshaped_j

    low_energy = torch.sum(x_low ** 2, dim=1)
    high_energy = torch.sum(x_high ** 2, dim=1)

    low_energy_reshaped_i = low_energy.view(-1, 1)
    high_energy_reshaped_j = high_energy.view(1, -1)
    denominator = low_energy.sum() + high_energy.sum()

    if denominator == 0:
        denominator = 1e-8

    filter_matrix = (low_energy_reshaped_i + high_energy_reshaped_j) / denominator

    attention_optimization_matrix = sum_matrix * filter_matrix

    attention_optimization_matrix = torch.nan_to_num(attention_optimization_matrix, nan=0.0)
    return attention_optimization_matrix


def compute_fourier_features(x, edge_index):

    num_nodes = x.shape[0]
    device = x.device

    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float64, device=device)
    adj[edge_index[0, :], edge_index[1, :]] = 1

    adj = adj + adj.t()
    adj = adj.fill_diagonal_(0)

    degree = torch.diag(adj.sum(dim=1))
    if torch.any(torch.diag(degree) == 0):
        epsilon = 1e-5
        eye_matrix = epsilon * torch.eye(degree.shape[0], device=device)
        d_sqrt = torch.inverse(torch.sqrt(degree + eye_matrix))
    else:
        d_sqrt = torch.inverse(torch.sqrt(degree))

    laplacian = torch.eye(num_nodes, device=device) - d_sqrt @ adj @ d_sqrt

    eigvals, eigvecs = torch.linalg.eig(laplacian)
    eigvals = eigvals.real.float()
    eigvecs = eigvecs.real.float()
    x_fourier = eigvecs.t() @ x.float()

    eigvals, idx = torch.sort(eigvals)
    eigvecs = eigvecs[:, idx]
    nodes = eigvecs.shape[0]
    k=int(0.2 * nodes)

    low_mask = torch.zeros_like(eigvals)
    low_mask[:k] = 1
    high_mask = 1 - low_mask

    x_low = eigvecs @ (x_fourier * low_mask.unsqueeze(1))
    x_high = eigvecs @ (x_fourier * high_mask.unsqueeze(1))

    attention_optimization_matrix = frequency_filtering(eigvals, x_low, x_high)

    return attention_optimization_matrix


def process_single_protein(sample, enable_fourier=True, enable_pe=True):
    x = torch.FloatTensor(sample['x'])
    y = sample['label']
    if type(y)==list:
        label = y
    elif type(y)==str:
        label=[int(c) for c in y]
    label=torch.LongTensor(label)
    sm, edge_index, _ = CosineSimilarity.compute_attention(x, threshold=0.7)
    processed_data = {
        'r_node': x,
        'residue_adj_t': sm.t(),
        'y': label
    }

    if enable_pe:
        # 计算位置编码
        k=config.PE_DIM
        graph_transformer = LapPE(k=k, attr_name='r_pe')
        r_pe_data = Data(x=x, edge_index=edge_index)
        r_pe_data = graph_transformer(r_pe_data)
        processed_data['r_pe'] = r_pe_data.r_pe

    if enable_fourier:
        # 计算傅里叶特征
        r_fourier = compute_fourier_features(x, edge_index)
        processed_data['r_fourier'] = r_fourier

    return processed_data


def preprocess_dataset(data_path, save_dir, enable_fourier=True, enable_pe=True, name: str = None):
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
        save_path = os.path.join(save_dir, f'{name}.pkl')
        torch.save(processed_data, save_path)

    print(f"预处理完成，结果保存在: {save_dir}")


if __name__ == "__main__":
    # 设置预处理参数
    ENABLE_FOURIER = True
    ENABLE_PE = True
    parser = argparse.ArgumentParser(description='预处理数据集')
    parser.add_argument('--data_path', required=True, help='数据文件路径')
    parser.add_argument('--name', required=True, help='数据集名称')

    args = parser.parse_args()
    # 预处理训练集
    preprocess_dataset(
        data_path=args.data_path,
        save_dir='/../gz-data/Test',
        enable_fourier=ENABLE_FOURIER,
        enable_pe=ENABLE_PE,
        name=args.name
    )
