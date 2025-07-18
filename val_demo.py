import os
import torch
import pickle
import argparse
from tqdm import tqdm

# 从我们的 utils 和 GASPPI 模块中导入必要的组件
from utils.metrics import calculate_metrics
from utils.find_best_thre import find_best_threshold_by_f_beta
# Import the new model and the PE utility
from GASPPI.model.dual_stream import DualStreamPPI
from GASPPI.utils import add_gaussian_edge_weights, add_laplacian_pe
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

def prepare_sample(sample, pe_dim, device):
    """
    Converts a single sample dictionary into a torch_geometric.data.Data object,
    computes Gaussian edge weights, and adds Laplacian Positional Encodings.
    (Matches the logic in demo.py)
    """
    # Create separate Data objects for atom and residue graphs to compute weights
    atom_graph_for_weights = Data(
        x=torch.FloatTensor(sample['atom_graph_node']),
        edge_index=torch.LongTensor(sample['atom_graph_edge']),
    )
    residue_graph_for_weights = Data(
        x=torch.FloatTensor(sample['residue_graph_node']),
        edge_index=torch.LongTensor(sample['residue_graph_edge']),
    )

    # Compute Gaussian edge weights
    atom_graph_for_weights = add_gaussian_edge_weights(atom_graph_for_weights, sigma=1.0)
    residue_graph_for_weights = add_gaussian_edge_weights(residue_graph_for_weights, sigma=1.0)

    # Create SparseTensors for model input
    atom_adj_t = SparseTensor(
        row=atom_graph_for_weights.edge_index[0], col=atom_graph_for_weights.edge_index[1],
        value=atom_graph_for_weights.edge_attr.squeeze(),
        sparse_sizes=(len(atom_graph_for_weights.x), len(atom_graph_for_weights.x))
    ).t()
    residue_adj_t = SparseTensor(
        row=residue_graph_for_weights.edge_index[0], col=residue_graph_for_weights.edge_index[1],
        value=residue_graph_for_weights.edge_attr.squeeze(),
        sparse_sizes=(len(residue_graph_for_weights.x), len(residue_graph_for_weights.x))
    ).t()

    # Now, create the main Data object for the DualStream model
    data = Data(
        atom_x=torch.FloatTensor(sample['atom_graph_node']),
        atom_adj_t=atom_adj_t,
        residue_x=torch.FloatTensor(sample['residue_graph_node']),
        residue_adj_t=residue_adj_t,
        edge_index=torch.LongTensor(sample['residue_graph_edge']), # Edges to predict
        y=torch.LongTensor(sample['label']),
        atom_to_residue_map=torch.tensor(sample['a2r_map'])
    )

    # Add Laplacian PE to the residue graph representation
    residue_graph_for_pe = Data(x=data.residue_x, edge_index=data.residue_adj_t.to_edge_index())
    residue_graph_for_pe = add_laplacian_pe(residue_graph_for_pe, pe_dim=pe_dim)
    data.lap_pe = residue_graph_for_pe.lap_pe

    return data.to(device)


@torch.no_grad()
def evaluate(model, data_list, pe_dim, device):
    """使用加载的单个模型在给定数据上进行评估。"""
    model.eval()
    all_probs = []
    all_targets = []

    print("开始模型评估...")
    for sample in tqdm(data_list, desc="Evaluating"):
        data = prepare_sample(sample, pe_dim, device)
        out = model(data) # Model now takes the full data object
        all_probs.append(torch.sigmoid(out))
        all_targets.append(data.y.float())

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
    
    # Geometric stream dimensions (must match demo.py)
    geo_hidden_dim = 128
    geo_out_dim = 64

    # Fusion dimension (must match demo.py)
    fusion_hidden_dim = 128

    model = DualStreamPPI(
        atom_in_channels=atom_in_channels,
        residue_in_channels=residue_in_channels,
        atom_hidden_dims=atom_hidden_dims,
        residue_hidden_dims=residue_hidden_dims,
        pe_dim=args.pe_dim,
        geo_hidden_dim=geo_hidden_dim,
        geo_out_dim=geo_out_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        out_channels=out_channels,
        heads=4,
        dropout=args.dropout
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
    final_metrics = evaluate(model, all_proteins, args.pe_dim, device)
    
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
    parser = argparse.ArgumentParser(description="评估单个DualStreamPPI模型在验证集上的性能")
    parser.add_argument('--full_data_path', type=str, required=True,
                        help='包含所有样本的.pkl文件路径，用于初始化模型和进行评估')
    parser.add_argument('--model_dir', type=str, default='./saved_models',
                        help='包含best_model.pth的目录')
    parser.add_argument('--seed', type=int, default=42, 
                        help='用于复现数据划分的随机种子 (必须与demo.py一致)')
    # 这个dropout值也必须与训练时使用的值一致
    parser.add_argument('--dropout', type=float, default=0.5, help='训练时使用的Dropout率')
    parser.add_argument('--pe_dim', type=int, default=16, help='训练时使用的Laplacian PE维度 (必须与demo.py一致)')
    
    args = parser.parse_args()
    main(args)