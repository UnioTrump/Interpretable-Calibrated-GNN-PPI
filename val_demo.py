import os
import torch
import pickle
import argparse
import numpy as np
import glob
from tqdm import tqdm

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
def evaluate_ensemble(models, data_list, device):
    """使用加载的一组模型在给定数据上进行集成评估。"""
    for model in models:
        model.eval()

    all_probs = []
    all_targets = []

    print("开始模型集成评估...")
    for sample in tqdm(data_list, desc="Ensemble Inference"):
        p_a_node, atom_adj_t, p_r_node, residue_adj_t, targets, a2r_map = prepare_sample(sample, device)

        # 存储来自每个模型的预测
        model_outputs = []
        for model in models:
            out = model(p_a_node, atom_adj_t, p_r_node, residue_adj_t, a2r_map)
            model_outputs.append(torch.sigmoid(out))

        # 对所有模型的预测结果取平均
        avg_probs = torch.stack(model_outputs).mean(dim=0)

        # 收集结果
        all_probs.append(avg_probs)
        all_targets.append(targets.float())

    all_targets_tensor = torch.cat(all_targets, dim=0).cpu()
    all_probs_tensor = torch.cat(all_probs, dim=0).cpu()

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
    # --- 1. 加载用于评估的测试数据 ---
    test_data = load_data(args.data_path)
    if not test_data:
        print("测试数据为空，无法进行评估。")
        return
        
    # --- 2. 搜索并加载所有K-Fold模型 ---
    model_paths = sorted(glob.glob(os.path.join(args.model_dir, 'fold_*/best_model.pth')))
    if not model_paths:
        print(f"在目录 '{args.model_dir}' 中没有找到任何 'fold_*/best_model.pth' 模型。")
        print("请确保您的模型目录结构正确，例如：./saved_models/fold_1/best_model.pth")
        return
    
    print(f"找到了 {len(model_paths)} 个模型用于集成:")
    for path in model_paths:
        print(f" - {path}")

    # 为了正确初始化模型大小，需要加载训练全集来获取最大节点数
    # 这是一个简化做法，更稳健的方式是保存训练时的配置
    print(f"从 {args.train_data_path} 加载训练数据以确定模型尺寸...")
    full_train_data = load_data(args.train_data_path)
    max_atom_nodes = max(len(d['atom_graph_node']) for d in full_train_data)
    max_residue_nodes = max(len(d['residue_graph_node']) for d in full_train_data)
    print(f"模型将使用 Max Atom Nodes={max_atom_nodes}, Max Residue Nodes={max_residue_nodes}进行初始化。")

    models = []
    for path in model_paths:
        # 基于新的、不含history的模型架构进行初始化
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
            dropout=0.4
        ).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        models.append(model)
    print(f"\n成功加载了 {len(models)} 个模型。")

    # --- 3. 执行集成评估 ---
    final_metrics = evaluate_ensemble(models, test_data, device)

    # --- 4. 打印评估报告 ---
    print("\n--- 模型集成评估报告 ---")
    print(f"评估数据集: {args.data_path}")
    print(f"模型来源目录: {args.model_dir}")
    print(f"集成了 {len(models)} 个模型")
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
    parser = argparse.ArgumentParser(description="使用K-Fold模型集成评估HierarchicalGNN性能")
    parser.add_argument('--data_path', type=str, default='/gz-data/Test60.pkl',
                        help='包含【测试数据】列表的pickle文件路径')
    parser.add_argument('--train_data_path', type=str, default='/gz-data/Train60.pkl',
                        help='包含【训练全集】的pickle文件路径 (用于初始化模型大小)')
    parser.add_argument('--model_dir', type=str, default='./saved_models',
                        help='包含K-Fold模型子目录(fold_*)的根目录')
    
    args = parser.parse_args()
    main(args)