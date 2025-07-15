import os
import torch
import pickle
import argparse
import numpy as np

# 从我们的 utils 和 GASPPI 模块中导入必要的组件
from utils.metrics import calculate_metrics
from utils.find_best_thre import find_best_threshold_by_f_beta
from GASPPI.model.PPI import HierarchicalGNN

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(pkl_path):
    """从pickle文件加载数据列表。"""
    print(f"从 {pkl_path} 加载数据...")
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
    # 但为了简化评估脚本，我们假设模型内部能处理edge_index
    # 或者如果模型需要SparseTensor，我们在这里创建它
    from torch_sparse import SparseTensor
    atom_edge_attr = torch.ones(p_a_edge.shape[1], dtype=torch.float)
    atom_adj_t = SparseTensor(
        row=p_a_edge[0], col=p_a_edge[1],
        value=atom_edge_attr,
        sparse_sizes=(len(p_a_node), len(p_a_node))
    ).t()
    residue_edge_attr = torch.ones(p_r_edge.shape[1], dtype=torch.float)
    residue_adj_t = SparseTensor(
        row=p_r_edge[0], col=p_r_edge[1],
        value=residue_edge_attr,
        sparse_sizes=(len(p_r_node), len(p_r_node))
    ).t()

    return (
        p_a_node.to(device), atom_adj_t.to(device),
        p_r_node.to(device), residue_adj_t.to(device),
        targets.to(device), a2r_map.to(device)
    )

@torch.no_grad()
def evaluate(model, data_list):
    """使用加载的模型在给定数据上进行评估。"""
    model.eval()
    all_probs = []
    all_targets = []

    print("开始模型评估...")
    for sample in data_list:
        p_a_node, atom_adj_t, p_r_node, residue_adj_t, targets, a2r_map = prepare_sample(sample, device)
        
        # 前向传播
        out = model(p_a_node, atom_adj_t, p_r_node, residue_adj_t, a2r_map)
        
        # 收集结果
        all_probs.append(torch.sigmoid(out))
        all_targets.append(targets.float())

    all_targets_tensor = torch.cat(all_targets, dim=0)
    all_probs_tensor = torch.cat(all_probs, dim=0)
    
    print("评估完成，开始计算指标...")
    # 寻找最佳阈值
    threshold, f_beta = find_best_threshold_by_f_beta(
        all_targets_tensor,
        all_probs_tensor,
        num_threshold=100
    )
    
    # 使用最佳阈值计算所有指标
    metrics = calculate_metrics(
        y_true=all_targets_tensor,
        y_scores=all_probs_tensor,
        threshold=threshold
    )
    
    return metrics

def main(args):
    # --- 1. 加载数据 ---
    # 注意：这里我们应该加载一个独立的测试集，或者用验证集来评估
    # 为了演示，我们先用与demo.py相同的分割方式
    full_data_list = load_data(args.data_path)
    np.random.seed(123) # 确保分割与训练时一致
    np.random.shuffle(full_data_list)
    split_num = int(0.8 * len(full_data_list))
    val_data = full_data_list[split_num:]

    # --- 2. 初始化模型 ---
    # 模型的参数必须与训练时保存的那个模型完全一致
    # 计算最大节点数以正确初始化模型
    max_atom_nodes = max(len(d['atom_graph_node']) for d in full_data_list)
    max_residue_nodes = max(len(d['residue_graph_node']) for d in full_data_list)
    
    model = HierarchicalGNN(
        atom_num_nodes=max_atom_nodes,
        residue_num_nodes=max_residue_nodes,
        atom_in_channels=37,
        residue_in_channels=1024,
        hidden_channels=256,
        out_channels=1,
        atom_num_layers=4,
        residue_num_layers=4,
        heads=4,
        dropout=0.4,
        pool_size=2,
        buffer_size=500,
        device=device
    ).to(device)

    # --- 3. 加载模型权重 ---
    print(f"从 {args.checkpoint_path} 加载模型权重...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("模型权重加载成功！")

    # --- 4. 执行评估 ---
    final_metrics = evaluate(model, val_data)

    # --- 5. 打印评估报告 ---
    print("\n--- 模型性能评估报告 ---")
    print(f"评估数据集: {args.data_path}")
    print(f"模型检查点: {args.checkpoint_path}")
    print("---------------------------------")
    print(f"最佳F-beta阈值: {final_metrics['threshold']:.4f}")
    print(f"PR AUC:          {final_metrics['pr_auc']:.4f}")
    print(f"ROC AUC:         {final_metrics['roc_auc']:.4f}")
    print("--- 在最佳阈值下的指标 ---")
    print(f"F1-Score:        {final_metrics['f1_score']:.4f}")
    print(f"Precision:       {final_metrics['precision']:.4f}")
    print(f"Recall:          {final_metrics['recall']:.4f}")
    print(f"Accuracy:        {final_metrics['accuracy']:.4f}")
    print(f"MCC:             {final_metrics['mcc']:.4f}")
    print("---------------------------------\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="评估HierarchicalGNN模型性能")
    parser.add_argument('--data_path', type=str, default='/gz-data/Test60.pkl',
                        help='包含数据列表的pickle文件路径')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/best_model_pr_auc.pt',
                        help='模型检查点文件路径')
    
    args = parser.parse_args()
    main(args)