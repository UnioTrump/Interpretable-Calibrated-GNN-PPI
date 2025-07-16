import os
import torch
import pickle
import argparse
from tqdm import tqdm

# 从我们的 utils 和 GASPPI 模块中导入必要的组件
from utils.metrics import calculate_metrics
from utils.find_best_thre import find_best_threshold_by_f_beta
from GASPPI.model.PPI import HierarchicalGNN

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(pkl_path):
    """从pickle文件加载数据列表。"""
    with open(pkl_path, 'rb') as f:
        data_list = pickle.load(f)
    print(f"加载了 {len(data_list)} 个样本。")
    return data_list

def prepare_sample(sample, device):
    """将单个样本数据转换为torch tensor并移动到指定设备。"""
    p_a_node = torch.FloatTensor(sample['atom_graph_node'])
    p_a_edge = torch.LongTensor(sample['atom_graph_edge'])
    p_r_node = torch.FloatTensor(sample['residue_graph_node'])
    p_r_edge = torch.LongTensor(sample['residue_graph_edge'])
    targets = torch.LongTensor(sample['label'])
    a2r_map = torch.tensor(sample['a2r_map'])

    # 这里的邻接矩阵创建逻辑需要与训练时一致
    from torch_sparse import SparseTensor
    atom_edge_attr = torch.ones(p_a_edge.shape[1], device=p_a_edge.device)
    atom_adj_t = SparseTensor(
        row=p_a_edge[0], col=p_a_edge[1], value=atom_edge_attr,
        sparse_sizes=(len(p_a_node), len(p_a_node))
    ).t()
    residue_edge_attr = torch.ones(p_r_edge.shape[1], device=p_r_edge.device)
    residue_adj_t = SparseTensor(
        row=p_r_edge[0], col=p_r_edge[1], value=residue_edge_attr,
        sparse_sizes=(len(p_r_node), len(p_r_node))
    ).t()

    return (
        p_a_node.to(device), atom_adj_t.to(device),
        p_r_node.to(device), residue_adj_t.to(device),
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
        p_a_node, atom_adj_t, p_r_node, residue_adj_t, targets, a2r_map = prepare_sample(sample, device)
        out = model(p_a_node, atom_adj_t, p_r_node, residue_adj_t, a2r_map)
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
    max_atom_nodes = max(len(d['atom_graph_node']) for d in all_proteins)
    max_residue_nodes = max(len(d['residue_graph_node']) for d in all_proteins)
    
    model = HierarchicalGNN(
        atom_num_nodes=max_atom_nodes,
        residue_num_nodes=max_residue_nodes,
        atom_in_channels=37,
        residue_in_channels=1024,
        hidden_channels=256,
        out_channels=1,
        atom_num_layers=3,      # 与demo.py同步
        residue_num_layers=3,   # 与demo.py同步
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
        # 将 "pr_auc" 这样的键转换为 "Pr Auc"
        formatted_key = key.replace('_', ' ').title()
        print(f"{formatted_key:<15}: {val:.4f}")
    print("---------------------------------------\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="评估单个HierarchicalGNN模型在验证集上的性能")
    parser.add_argument('--full_data_path', type=str, required=True,
                        help='测试集')
    parser.add_argument('--model_dir', type=str, default='./saved_models',
                        help='包含best_model.pth的目录')
    parser.add_argument('--seed', type=int, default=42, 
                        help='用于复现数据划分的随机种子 (必须与demo.py一致)')
    # 这个dropout值也必须与训练时使用的值一致
    parser.add_argument('--dropout', type=float, default=0.5, help='训练时使用的Dropout率')
    
    args = parser.parse_args()
    main(args)