import os
import torch
import pickle
import argparse
from tqdm import tqdm

# 从我们的 utils 和 GASPPI 模块中导入必要的组件
from utils.metrics import calculate_metrics
from utils.find_best_thre import find_best_threshold_by_f_beta
from GASPPI.model.PPI import ProteinGNN
from GASPPI.utils import add_gaussian_edge_weights
from torch_geometric.data import Data
from torch_sparse import SparseTensor

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(pkl_path):
    """从pickle文件加载数据列表。"""
    with open(pkl_path, 'rb') as f:
        data_list = pickle.load(f)
    print(f"加载了 {len(data_list)} 个样本。")
    return data_list

def prepare_sample(sample, device):
    """将单个样本数据转换为torch_geometric.data.Data对象，并计算边权重。"""
    atom_graph = Data(
        x=torch.FloatTensor(sample['atom_graph_node']),
        edge_index=torch.LongTensor(sample['atom_graph_edge']),
    )
    residue_graph = Data(
        x=torch.FloatTensor(sample['residue_graph_node']),
        edge_index=torch.LongTensor(sample['residue_graph_edge']),
    )

    # 计算高斯边权重
    atom_graph = add_gaussian_edge_weights(atom_graph, sigma=1.0)
    residue_graph = add_gaussian_edge_weights(residue_graph, sigma=1.0)

    # 创建稀疏邻接矩阵，同时包含【结构】和【权重】
    atom_adj_t = SparseTensor(
        row=atom_graph.edge_index[0], col=atom_graph.edge_index[1],
        value=atom_graph.edge_attr,  # 保持 [N, 1] 形状
        sparse_sizes=(len(atom_graph.x), len(atom_graph.x))
    ).t()
    residue_adj_t = SparseTensor(
        row=residue_graph.edge_index[0], col=residue_graph.edge_index[1],
        value=residue_graph.edge_attr, # 保持 [N, 1] 形状
        sparse_sizes=(len(residue_graph.x), len(residue_graph.x))
    ).t()

    targets = torch.LongTensor(sample['label'])
    a2r_map = torch.tensor(sample['a2r_map'])

    return (
        atom_graph.x.to(device), atom_adj_t.to(device),
        residue_graph.x.to(device), residue_adj_t.to(device),
        targets.to(device), a2r_map.to(device)
    )

@torch.no_grad()
def evaluate(model, data_list, device):
    """使用加载的单个模型在给定数据上进行评估。"""
    model.eval()
    all_probs = []
    all_targets = []

    print("开始模型评估...")
    for sample in tqdm(data_list, desc="Evaluating"):
        atom_graph_x, atom_adj_t, residue_graph_x, residue_adj_t, targets, a2r_map = prepare_sample(sample, device)
        out = model(atom_graph_x, atom_adj_t, residue_graph_x, residue_adj_t, a2r_map)
        all_probs.append(torch.sigmoid(out))
        all_targets.append(targets.float())

    all_targets_tensor = torch.cat(all_targets, dim=0).cpu()
    all_probs_tensor = torch.cat(all_probs, dim=0).cpu()

    print("评估完成，开始计算指标...")
    threshold, _ = find_best_threshold_by_f_beta(all_targets_tensor, all_probs_tensor, num_threshold=100)
    metrics = calculate_metrics(y_true=all_targets_tensor, y_scores=all_probs_tensor, threshold=threshold)
    
    return metrics

def main(args):
    # --- 1. 加载并划分数据，与demo.py保持完全一致 ---
    # 无论是评估验证集还是独立的测试集，都需要加载训练集来确定模型大小
    print(f"从 {args.full_data_path} 加载全集数据以确保划分和模型尺寸一致...")
    all_proteins = load_data(args.full_data_path)
    if not all_proteins:
        print("数据列表为空，无法进行。")
        return

    # --- 2. 初始化模型，与demo.py保持完全一致 ---
    sample_data = all_proteins[0]
    atom_in_channels = sample_data['atom_graph_node'].shape[1]
    residue_in_channels = sample_data['residue_graph_node'].shape[1]
    out_channels = 1

    atom_hidden_dims = [128, 256, 128]
    residue_hidden_dims = [256, 512, 256, 128]
    
    model = ProteinGNN(
        atom_in_channels=atom_in_channels,
        residue_in_channels=residue_in_channels,
        atom_hidden_dims=atom_hidden_dims,
        residue_hidden_dims=residue_hidden_dims,
        out_channels=out_channels,
        heads=4,
        dropout=args.dropout # 与demo.py同步
    ).to(device)

    # --- 3. 加载训练好的模型权重 ---
    model_path = os.path.join(args.model_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"错误：在 '{model_path}' 未找到模型文件。请先运行demo.py进行训练。")
        return
        
    print(f"从 {model_path} 加载模型权重...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("模型权重加载成功！")

    # --- 4. 执行评估 ---
    final_metrics = evaluate(model, all_proteins, device)
    
    # --- 5. 打印评估报告 ---
    print("\n--- 模型性能评估报告 (验证集) ---")
    print("---------------------------------------")
    for key, val in final_metrics.items():
        if isinstance(val, (int, float)):
             print(f"  {key.replace('_', ' ').title()}: {val:.4f}")
        elif isinstance(val, dict):
            print(f"  {key.replace('_', ' ').title()}:")
            for sub_key, sub_val in val.items():
                print(f"    {sub_key.upper()}: {sub_val}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {val}")
    print("---------------------------------------\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="评估单个ProteinGNN模型在验证集上的性能")
    parser.add_argument('--full_data_path', type=str, required=True,
                        help='包含所有样本的.pkl文件路径，用于初始化模型和进行评估')
    parser.add_argument('--model_dir', type=str, default='./saved_models',
                        help='包含best_model.pth的目录')
    parser.add_argument('--seed', type=int, default=42, 
                        help='用于复现数据划分的随机种子 (必须与demo.py一致)')
    # 这个dropout值也必须与训练时使用的值一致
    parser.add_argument('--dropout', type=float, default=0.5, help='训练时使用的Dropout率')
    
    args = parser.parse_args()
    main(args)