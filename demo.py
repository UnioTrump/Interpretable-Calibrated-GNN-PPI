from tqdm import tqdm
import pickle
import torch
import os
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_sparse import SparseTensor
from utils.losses import WeightedCrossEntropy
from utils.metrics import calculate_metrics
from utils.find_best_thre import find_best_threshold_by_f_beta
from GASPPI.model.PPI import ProteinGNN
from GASPPI.utils import add_gaussian_edge_weights
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)


def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        raw_data = pickle.load(f)
    return raw_data


def prepare_sample(sample, device):
    """将单个样本数据转换为torch_geometric.data.Data对象，并计算边权重"""
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
        value=atom_graph.edge_attr, # 移除 .squeeze()
        sparse_sizes=(len(atom_graph.x), len(atom_graph.x))
    ).t()

    residue_adj_t = SparseTensor(
        row=residue_graph.edge_index[0], col=residue_graph.edge_index[1],
        value=residue_graph.edge_attr, # 移除 .squeeze()
        sparse_sizes=(len(residue_graph.x), len(residue_graph.x))
    ).t()

    targets = torch.LongTensor(sample['label'])
    a2r_map = torch.tensor(sample['a2r_map'])

    return (
        atom_graph.x.to(device), atom_adj_t.to(device),
        residue_graph.x.to(device), residue_adj_t.to(device),
        targets.to(device), a2r_map.to(device)
    )


def train(model, train_proteins, optimizer, batch_size, pos_weight, grad_norm=None):
    model.train()
    np.random.shuffle(train_proteins)

    total_loss = 0
    criterion = WeightedCrossEntropy(pos_wt=pos_weight, device=device)

    for i in range(0, len(train_proteins), batch_size):
        batch_proteins = train_proteins[i:i + batch_size]
        optimizer.zero_grad()

        batch_loss_sum = 0
        for protein in batch_proteins:
            p_a_node, atom_adj_t, p_r_node, residue_adj_t, targets, a2r_map = prepare_sample(protein, device)
            out = model(p_a_node, atom_adj_t, p_r_node, residue_adj_t, a2r_map)
            loss = criterion.compute_loss(out, targets)
            batch_loss_sum += loss.item()
            loss = loss / len(batch_proteins)
            loss.backward()

        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)

        optimizer.step()
        total_loss += batch_loss_sum

    return total_loss / len(train_proteins)


@torch.no_grad()
def test(model, val_proteins, pos_weight):
    model.eval()
    criterion = WeightedCrossEntropy(pos_wt=pos_weight, device=device)
    total_loss = 0
    all_probs, all_targets = [], []

    for val_p in val_proteins:
        p_a_node, atom_adj_t, p_r_node, residue_adj_t, targets, a2r_map = prepare_sample(val_p, device)
        out = model(p_a_node, atom_adj_t, p_r_node, residue_adj_t, a2r_map)
        loss = criterion.compute_loss(out, targets)
        total_loss += loss.item()
        all_probs.append(torch.sigmoid(out))
        all_targets.append(targets.float())

    avg_loss = total_loss / len(val_proteins)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    all_probs_tensor = torch.cat(all_probs, dim=0)

    threshold, _ = find_best_threshold_by_f_beta(all_targets_tensor, all_probs_tensor, num_threshold=100)
    metrics = calculate_metrics(y_true=all_targets_tensor, y_scores=all_probs_tensor, threshold=threshold)

    return avg_loss, metrics, threshold

def plot_loss_curves(train_losses, val_losses, save_path='loss_curves.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if val_losses:
        best_val_epoch = np.argmin(val_losses)
        best_val_loss = val_losses[best_val_epoch]
        plt.axvline(x=best_val_epoch + 1, color='gray', linestyle='--')
        plt.scatter(best_val_epoch + 1, best_val_loss, marker='*', s=150, color='gold', zorder=5, label=f'Best Val Loss: {best_val_loss:.4f} at Epoch {best_val_epoch+1}')
    
    plt.legend()
    plt.savefig(save_path, dpi=300)
    print(f"Loss curve saved to: {save_path}")
    plt.close()

def main(args):
    POS_WEIGHT = torch.tensor(args.pos_weight, device=device)
    
    # --- 1. 加载并划分数据集 ---
    print("加载并划分数据...")
    all_proteins = load_data(args.data_path)
    if not all_proteins:
        print("Error: Data list is empty.")
        return

    # 设置随机种子以保证每次划分一致
    np.random.seed(args.seed)
    np.random.shuffle(all_proteins)
    
    split_index = int(len(all_proteins) * 0.8)
    train_data = all_proteins[:split_index]
    val_data   = all_proteins[split_index:]
    print('train_data', len(train_data))
    print('val_data', len(val_data))

    # --- 2. 初始化模型 ---
    # 从第一个数据样本中动态获取维度信息
    sample_data = all_proteins[0]
    atom_in_channels = sample_data['atom_graph_node'].shape[1]
    residue_in_channels = sample_data['residue_graph_node'].shape[1]
    out_channels = 1 # 二分类

    # 定义具有层次结构的隐藏维度
    atom_hidden_dims = [128, 256, 128]
    residue_hidden_dims = [256, 512, 256, 128]

    model = ProteinGNN(
        atom_in_channels=atom_in_channels,
        residue_in_channels=residue_in_channels,
        atom_hidden_dims=atom_hidden_dims,
        residue_hidden_dims=residue_hidden_dims,
        out_channels=out_channels,
        dropout=args.dropout,
        heads=4
    ).to(device)
    print("模型已成功初始化 (ProteinGNN):")
    print(model)

    # --- 3. 设置优化器和调度器 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience // 2, verbose=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs // 2, eta_min=1e-6)


    best_pr_auc = 0.0
    patience_counter = 0
    os.makedirs(args.model_dir, exist_ok=True)
    best_model_path = os.path.join(args.model_dir, 'best_model.pth')
    
    # --- 4. 开始训练循环 ---
    print("开始训练...")
    train_losses, val_losses = [], []
    epoch_pbar = tqdm(range(args.epochs), desc="Training Progress", ncols=180)

    for epoch in epoch_pbar:
        train_loss = train(model, train_data, optimizer, args.batch_size, POS_WEIGHT, grad_norm=1.0)
        train_losses.append(train_loss)

        val_loss, metrics, best_threshold = test(model, val_data, POS_WEIGHT)
        val_losses.append(val_loss)
        
        pr_auc = metrics['pr_auc']
        scheduler.step()

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"Epoch {epoch+1}: New best model saved with PR_AUC: {pr_auc:.4f}")
        else:
            patience_counter += 1

        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}',
            'Val PR_AUC': f'{pr_auc:.4f}', 'Val ROC_AUC': f'{metrics["roc_auc"]:.4f}',
            'Patience': f'{patience_counter}/{args.patience}'
        })

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    # --- 5. 评估并报告最终结果 ---
    print("\n--- 训练结束 ---")
    print(f"加载最佳模型 '{best_model_path}' 进行最终评估...")
    model.load_state_dict(torch.load(best_model_path))
    final_val_loss, final_metrics, final_threshold = test(model, val_data, POS_WEIGHT)

    print("\n--- 最终模型性能 (验证集) ---")
    for key, val in final_metrics.items():
        if isinstance(val, (int, float)):
             print(f"  {key.replace('_', ' ').title()}: {val:.4f}")
        elif isinstance(val, dict):
            print(f"  {key.replace('_', ' ').title()}:")
            for sub_key, sub_val in val.items():
                print(f"    {sub_key.upper()}: {sub_val}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {val}")


    # --- 6. 绘制损失曲线 ---
    plot_save_path = os.path.join(args.plot_dir, 'loss_curve.png')
    plot_loss_curves(train_losses, val_losses, save_path=plot_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ProteinGNN for PPI using a fixed 80/20 split')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data pkl file')
    parser.add_argument('--model_dir', type=str, default='./saved_models', help='Directory to save models')
    parser.add_argument('--plot_dir', type=str, default='./plots', help='Directory to save loss plots')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data splitting')

    parser.add_argument('--lr', type=float, default=4e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=4e-4, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.7, help='Dropout rate')
    
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for gradient accumulation')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--pos_weight', type=float, default=1.0, help='Positive weight for BCE loss')
    
    args = parser.parse_args()
    main(args)