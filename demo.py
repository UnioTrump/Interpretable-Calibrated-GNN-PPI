from tqdm import tqdm
import pickle
import torch
import os
import time
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_sparse import SparseTensor
from utils.losses import WeightedCrossEntropy
from utils.metrics import calculate_metrics
from utils.find_best_thre import find_best_threshold_by_f_beta
from GASPPI.model.PPI import HierarchicalGNN
import matplotlib.pyplot as plt
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(123)


def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        raw_data = pickle.load(f)
    return raw_data


def prepare_sample(sample, device):
    """将单个样本数据转换为torch tensor并移动到指定设备"""
    p_a_node = torch.FloatTensor(sample['atom_graph_node'])
    p_a_edge = torch.LongTensor(sample['atom_graph_edge'])
    p_r_node = torch.FloatTensor(sample['residue_graph_node'])
    p_r_edge = torch.LongTensor(sample['residue_graph_edge'])
    targets = torch.LongTensor(sample['label'])
    a2r_map = torch.tensor(sample['a2r_map'])

    atom_adj_t = SparseTensor(
        row=p_a_edge[0], col=p_a_edge[1],
        sparse_sizes=(len(p_a_node), len(p_a_node))
    ).t()

    residue_adj_t = SparseTensor(
        row=p_r_edge[0], col=p_r_edge[1],
        sparse_sizes=(len(p_r_node), len(p_r_node))
    ).t()

    return (
        p_a_node.to(device), atom_adj_t.to(device),
        p_r_node.to(device), residue_adj_t.to(device),
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
    
    all_proteins = load_data(args.data_path)
    if not all_proteins:
        print("Error: Data list is empty.")
        return

    np.random.shuffle(all_proteins)
    
    k_folds = args.k_folds
    fold_size = len(all_proteins) // k_folds
    folds = [all_proteins[i*fold_size:(i+1)*fold_size] for i in range(k_folds)]
    if len(all_proteins) % k_folds != 0:
        # Add remaining samples to the last fold
        folds[-1].extend(all_proteins[k_folds*fold_size:])

    all_fold_metrics = []

    for fold_idx in range(k_folds):
        print(f"--- Starting Fold {fold_idx+1}/{k_folds} ---")

        val_data = folds[fold_idx]
        train_data = []
        for i in range(k_folds):
            if i != fold_idx:
                train_data.extend(folds[i])

        atom_nodes = [len(d['atom_graph_node']) for d in all_proteins]
        residue_nodes = [len(d['residue_graph_node']) for d in all_proteins]
        max_atom_nodes = max(atom_nodes) if atom_nodes else 0
        max_residue_nodes = max(residue_nodes) if residue_nodes else 0

        model = HierarchicalGNN(
            atom_num_nodes=max_atom_nodes,
            residue_num_nodes=max_residue_nodes,
            atom_in_channels=37,
            residue_in_channels=1024,
            hidden_channels=256,
            out_channels=1,
            atom_num_layers=4,
            residue_num_layers=4,
            dropout=0.4,
            heads=4
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)

        best_pr_auc = 0.0
        patience_counter = 0
        fold_model_dir = os.path.join(args.model_dir, f'fold_{fold_idx+1}')
        os.makedirs(fold_model_dir, exist_ok=True)
        
        train_losses, val_losses = [], []
        epoch_pbar = tqdm(range(args.epochs), desc=f"Fold {fold_idx+1} Progress", ncols=180)

        for epoch in epoch_pbar:
            train_loss = train(model, train_data, optimizer, args.batch_size, POS_WEIGHT, grad_norm=1.0)
            train_losses.append(train_loss)

            val_loss, metrics, best_threshold = test(model, val_data, POS_WEIGHT)
            val_losses.append(val_loss)
            
            pr_auc = metrics['pr_auc']

            scheduler.step(val_loss)

            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(fold_model_dir, 'best_model.pth'))
            else:
                patience_counter += 1

            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}',
                'Val PR_AUC': f'{pr_auc:.4f}', 'roc_auc': f'{metrics["roc_auc"]:.4f}',
                'Patience': f'{patience_counter}/{args.patience}'
            })

            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Load best model for final evaluation on this fold
        model.load_state_dict(torch.load(os.path.join(fold_model_dir, 'best_model.pth')))
        _, final_metrics, _ = test(model, val_data, POS_WEIGHT)
        all_fold_metrics.append(final_metrics)

        plot_save_path = os.path.join(args.plot_dir, f'loss_curve_fold_{fold_idx+1}.png')
        plot_loss_curves(train_losses, val_losses, save_path=plot_save_path)

    print("\n--- K-Fold Cross-Validation Finished ---")
    
    # Calculate and print average metrics
    avg_metrics = {key: np.mean([m[key] for m in all_fold_metrics]) for key in all_fold_metrics[0]}
    std_metrics = {key: np.std([m[key] for m in all_fold_metrics]) for key in all_fold_metrics[0]}

    print("Average metrics over all folds:")
    for key, val in avg_metrics.items():
        print(f"  {key}: {val:.4f} (+/- {std_metrics[key]:.4f})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train HierarchicalGNN for PPI')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data pkl file')
    parser.add_argument('--model_dir', type=str, default='./saved_models', help='Directory to save models')
    parser.add_argument('--plot_dir', type=str, default='./plots', help='Directory to save loss plots')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for gradient accumulation')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--pos_weight', type=float, default=2.0, help='Positive weight for BCE loss')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for cross-validation')
    
    args = parser.parse_args()
    main(args)