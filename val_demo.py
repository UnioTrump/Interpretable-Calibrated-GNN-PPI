import os
import torch
import pickle
from tqdm import tqdm

from utils import calculate_metrics, find_best_threshold_by_f_beta
from GASPPI import add_laplacian_pe, H_GNNMambaPPI
from torch_geometric.data import Data
from torch_sparse import SparseTensor
import config

device = config.DEVICE

def load_data(pkl_path):
    """直接加载单个pkl文件进行验证"""
    print(f"从 '{pkl_path}' 加载验证数据...")
    if not os.path.isfile(pkl_path):
        print(f"错误: 文件不存在 - {pkl_path}")
        raise FileNotFoundError(f"未找到指定的验证数据文件: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        data_list = pickle.load(f)
    print(f"Loaded {len(data_list)} samples for evaluation from {pkl_path}.")
    return data_list

def prepare_sample(sample, device):
    """Prepares a single protein sample for the H_GNNMambaPPI model."""
    
    # Atom-level graph
    # 使用 torch.as_tensor 替换 torch.tensor 来彻底解决 UserWarning
    row_a = torch.as_tensor(sample['a_edge_index'][0], dtype=torch.long)
    col_a = torch.as_tensor(sample['a_edge_index'][1], dtype=torch.long)
    atom_adj_t = SparseTensor(
        row=row_a,
        col=col_a,
        sparse_sizes=(len(sample['a_node']), len(sample['a_node']))
    ).t()
    
    atom_edge_attr = None
    if 'a_edge_feat' in sample:
        atom_edge_attr = torch.as_tensor(sample['a_edge_feat'], dtype=torch.float)
        atom_edge_attr = atom_edge_attr / config.ATOM_DISTANCE_THRESHOLD

    # Residue-level graph
    row_r = torch.as_tensor(sample['r_edge_index'][0], dtype=torch.long)
    col_r = torch.as_tensor(sample['r_edge_index'][1], dtype=torch.long)
    residue_adj_t = SparseTensor(
        row=row_r,
        col=col_r,
        sparse_sizes=(len(sample['r_node']), len(sample['r_node']))
    ).t()
    
    residue_edge_attr = None
    if 'r_edge_feat' in sample:
        residue_edge_attr = torch.as_tensor(sample['r_edge_feat'], dtype=torch.float)
        residue_edge_attr = residue_edge_attr / config.RESIDUE_DISTANCE_THRESHOLD

    label_list = [int(char) for char in sample['label']]
    label_tensor = torch.as_tensor(label_list, dtype=torch.float)

    data = Data(
        atom_x=torch.as_tensor(sample['a_node'], dtype=torch.float),
        atom_adj_t=atom_adj_t,
        atom_edge_attr=atom_edge_attr,
        atom_to_residue_map=torch.as_tensor(sample['a2r_map'], dtype=torch.long),
        residue_x=torch.as_tensor(sample['r_node'], dtype=torch.float),
        residue_adj_t=residue_adj_t,
        residue_edge_attr=residue_edge_attr,
        y=label_tensor,
    )

    # Add Laplacian Positional Encodings for residue graph
    pe_edge_index = torch.as_tensor(sample['r_edge_index'], dtype=torch.long)
    residue_graph_for_pe = Data(num_nodes=data.residue_x.size(0), edge_index=pe_edge_index.contiguous())
    residue_graph_for_pe = add_laplacian_pe(residue_graph_for_pe, pe_dim=config.PE_DIM)
    data.lap_pe = residue_graph_for_pe.lap_pe

    return data.to(device)


@torch.no_grad()
def evaluate(model, data_list, device):
    model.eval()
    all_probs = []
    all_targets = []
    skipped_count = 0

    print("Starting model evaluation...")
    for sample in tqdm(data_list, desc="Evaluating"):
        try:
            data = prepare_sample(sample, device)
            out = model(data)
            all_probs.append(torch.sigmoid(out))
            all_targets.append(data.y.float())
        except RuntimeError as e:
            # 捕获由数据不一致（张量尺寸不匹配）导致的运行时错误
            if "Sizes of tensors must match" in str(e):
                print(f"\n[警告] 检测到数据不一致，跳过评估样本 (PID: {sample.get('PID', 'N/A')})。")
                skipped_count += 1
                continue
            else:
                # 如果是其它运行时错误，则重新抛出
                raise e

    # 如果所有样本都被跳过，则返回一个虚拟指标以避免崩溃
    if len(data_list) == skipped_count:
        print("\n[错误] 所有评估样本都因不一致而被跳过，无法计算指标。")
        return {'pr_auc': 0.0, 'roc_auc': 0.0, 'f1': 0.0, 'recall': 0.0, 'precision': 0.0, 'accuracy': 0.0}

    all_targets_tensor = torch.cat(all_targets, dim=0).cpu()
    all_probs_tensor = torch.cat(all_probs, dim=0).cpu()

    print("Evaluation complete, calculating metrics...")
    threshold, _ = find_best_threshold_by_f_beta(all_targets_tensor, all_probs_tensor, num_threshold=100)
    metrics = calculate_metrics(y_true=all_targets_tensor, y_scores=all_probs_tensor, threshold=threshold)
    
    return metrics

def main():
    # 加载验证数据集
    val_data = load_data(config.VAL_DATA_PATH)
    if not val_data:
        print("Validation data list is empty. Exiting.")
        return

    # Use the first sample to determine model dimensions
    sample_data = val_data[0]
    atom_in_channels = sample_data['a_node'].shape[1]
    residue_in_channels = sample_data['r_node'].shape[1]
    atom_edge_dim = sample_data['a_edge_feat'].shape[1]
    residue_edge_dim = sample_data['r_edge_feat'].shape[1]

    model_class = H_GNNMambaPPI
    print(f"--- Using Hierarchical SOTA Model for Validation: {model_class.__name__} ---")

    model = model_class(
        atom_in_channels=atom_in_channels,
        residue_in_channels=residue_in_channels,
        pe_dim=config.PE_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_atom_layers=config.NUM_ATOM_LAYERS,
        num_residue_layers=config.NUM_RESIDUE_LAYERS,
        out_channels=config.OUT_CHANNELS,
        mamba_d_state=config.MAMBA_D_STATE,
        mamba_d_conv=config.MAMBA_D_CONV,
        mamba_expand=config.MAMBA_EXPAND,
        dropout=config.DROPOUT,
        heads=config.HEADS,
        atom_edge_dim=atom_edge_dim,
        residue_edge_dim=residue_edge_dim
    ).to(device)

    model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'. Please run demo.py first.")
        return
        
    print(f"Loading model weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model weights loaded successfully!")

    final_metrics = evaluate(model, val_data, device)
    
    print("\n--- Model Performance Report (Validation Set) ---")
    print("-------------------------------------------------")
    for key, val in final_metrics.items():
        if isinstance(val, (int, float)):
             print(f"  {key.replace('_', ' ').title()}: {val:.4f}")
        elif isinstance(val, dict):
            print(f"  {key.replace('_', ' ').title()}:")
            for sub_key, sub_val in val.items():
                print(f"    {sub_key.upper()}: {sub_val}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {val}")
    print("-------------------------------------------------\n")

if __name__ == '__main__':
    main()