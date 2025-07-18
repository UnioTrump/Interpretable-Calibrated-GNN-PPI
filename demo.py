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
# Import the new model and the PE utility
from GASPPI.model.dual_stream import DualStreamPPI
from GASPPI.utils import add_gaussian_edge_weights, add_laplacian_pe
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

def load_data(pkl_path):
    if os.path.isdir(pkl_path):
        return sum((pickle.load(open(os.path.join(pkl_path, f), 'rb'))
                   for f in os.listdir(pkl_path) if f.endswith('.pkl')), [])
    return pickle.load(open(pkl_path, 'rb'))


def prepare_sample(sample, pe_dim, device):
    """
    Converts a single sample dictionary into a torch_geometric.data.Data object,
    computes Gaussian edge weights, and adds Laplacian Positional Encodings.
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
    # We create a temporary data object for this, as the PE is based on residue connectivity
    residue_graph_for_pe = Data(x=data.residue_x, edge_index=data.residue_adj_t.to_edge_index())
    residue_graph_for_pe = add_laplacian_pe(residue_graph_for_pe, pe_dim=pe_dim)
    data.lap_pe = residue_graph_for_pe.lap_pe

    return data.to(device)


def train(model, train_proteins, optimizer, batch_size, pos_weight, pe_dim, grad_norm=None):
    model.train()
    np.random.shuffle(train_proteins)

    total_loss = 0
    criterion = WeightedCrossEntropy(pos_wt=pos_weight, device=device)

    for i in range(0, len(train_proteins), batch_size):
        batch_proteins = train_proteins[i:i + batch_size]
        optimizer.zero_grad()

        batch_loss_sum = 0
        for protein in batch_proteins:
            data = prepare_sample(protein, pe_dim, device)
            out = model(data) # Model now takes the full data object
            loss = criterion.compute_loss(out, data.y)
            batch_loss_sum += loss.item()
            loss = loss / len(batch_proteins)
            loss.backward()

        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)

        optimizer.step()
        total_loss += batch_loss_sum

    return total_loss / len(train_proteins)


@torch.no_grad()
def test(model, val_proteins, pos_weight, pe_dim):
    model.eval()
    criterion = WeightedCrossEntropy(pos_wt=pos_weight, device=device)
    total_loss = 0
    all_probs, all_targets = [], []

    for val_p in val_proteins:
        data = prepare_sample(val_p, pe_dim, device)
        out = model(data) # Model now takes the full data object
        loss = criterion.compute_loss(out, data.y)
        total_loss += loss.item()
        all_probs.append(torch.sigmoid(out))
        all_targets.append(data.y.float())

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
    PE_DIM = args.pe_dim # Get PE dimension from args

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

    # Geometric stream dimensions
    geo_hidden_dim = 128
    geo_out_dim = 64

    # Fusion dimension
    fusion_hidden_dim = 128

    model = DualStreamPPI(
        atom_in_channels=atom_in_channels,
        residue_in_channels=residue_in_channels,
        atom_hidden_dims=atom_hidden_dims,
        residue_hidden_dims=residue_hidden_dims,
        pe_dim=PE_DIM,
        geo_hidden_dim=geo_hidden_dim,
        geo_out_dim=geo_out_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        out_channels=out_channels,
        dropout=args.dropout,
        heads=4
    ).to(device)
    print("模型已成功初始化 (DualStreamPPI):")
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
        train_loss = train(model, train_data, optimizer, args.batch_size, POS_WEIGHT, pe_dim=PE_DIM, grad_norm=1.0)
        train_losses.append(train_loss)

        val_loss, metrics, best_threshold = test(model, val_data, POS_WEIGHT, pe_dim=PE_DIM)
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
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for gradient accumulation')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--pos_weight', type=float, default=1.0, help='Positive weight for BCE loss')
    parser.add_argument('--pe_dim', type=int, default=16, help='Dimension of Laplacian Positional Encodings')
    
    args = parser.parse_args()
    main(args)