from tqdm import tqdm
import torch
import os
import time
import numpy as np
import collections
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from utils.losses import WeightedCrossEntropy
from utils.metrics import calculate_metrics
from utils.find_best_thre import find_best_threshold_by_f_beta
from utils.data import load_dataset
from GASPPI.model.PPI import HierarchicalGNN
import matplotlib.pyplot as plt
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, loader, optimizer, criterion, device, grad_norm=None):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion.compute_loss(out, batch.y)
        loss.backward()
        
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
            
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)

@torch.no_grad()
def test(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs, all_targets = [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion.compute_loss(out, batch.y)
        total_loss += loss.item() * batch.num_graphs
        all_probs.append(torch.sigmoid(out))
        all_targets.append(batch.y)

    avg_loss = total_loss / len(loader.dataset)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    all_probs_tensor = torch.cat(all_probs, dim=0)

    threshold, _ = find_best_threshold_by_f_beta(all_targets_tensor, all_probs_tensor, num_threshold=100)
    metrics = calculate_metrics(y_true=all_targets_tensor, y_scores=all_probs_tensor, threshold=threshold)

    return avg_loss, metrics, threshold

def plot_loss_curves(train_losses, val_losses, fold, save_path='loss_curves'):
    os.makedirs(save_path, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
    plt.title(f'Fold {fold + 1} - Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if val_losses:
        best_val_epoch = np.argmin(val_losses)
        best_val_loss = val_losses[best_val_epoch]
        plt.axvline(x=best_val_epoch + 1, color='gray', linestyle='--', label=f'Best Val Loss Epoch: {best_val_epoch+1}')
        plt.scatter(best_val_epoch + 1, best_val_loss, marker='*', s=150, color='gold', zorder=5, label=f'Best Val Loss: {best_val_loss:.4f}')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = collections.OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    save_filename = os.path.join(save_path, f'fold_{fold+1}_loss_curve.png')
    plt.savefig(save_filename, dpi=300)
    plt.close()

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("Loading and processing dataset...")
    dataset = load_dataset(data_path=args.data_path)
    print("Dataset loaded successfully.")

    # Squeeze labels to be 1D for StratifiedKFold
    labels = np.array([data.y.item() for data in dataset])
    
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels)):
        print(f"\n===== Fold {fold + 1}/{args.n_splits} =====")
        
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        
        train_labels = labels[train_idx]
        num_pos = np.sum(train_labels == 1)
        num_neg = np.sum(train_labels == 0)
        pos_weight_value = num_neg / num_pos if num_pos > 0 else 1.0
        pos_weight_tensor = torch.tensor(pos_weight_value, device=device)
        criterion = WeightedCrossEntropy(pos_wt=pos_weight_tensor, device=device)
        print(f"Fold {fold+1} POS_WEIGHT: {pos_weight_value:.2f}")

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

        data_sample = dataset[0]
        model = HierarchicalGNN(
            atom_in_channels=data_sample.atom_x.size(1),
            residue_in_channels=data_sample.residue_x.size(1),
            hidden_channels=args.hidden_channels,
            out_channels=1,
            atom_num_layers=args.atom_num_layers,
            residue_num_layers=args.residue_num_layers,
            dropout=args.dropout,
            heads=args.heads
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.scheduler_patience, verbose=True)

        best_pr_auc = 0.0
        patience_counter = 0
        best_epoch_metrics = {}
        model_dir = os.path.join(args.model_dir, f"fold_{fold+1}")
        os.makedirs(model_dir, exist_ok=True)
        
        train_losses, val_losses = [], []
        epoch_pbar = tqdm(range(args.epochs), desc=f"Fold {fold+1} Training", ncols=180)

        for epoch in epoch_pbar:
            train_loss = train(model, train_loader, optimizer, criterion, device, grad_norm=1.0)
            train_losses.append(train_loss)

            val_loss, metrics, best_threshold = test(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            
            pr_auc = metrics['pr_auc']
            scheduler.step(pr_auc)

            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                best_epoch_metrics = metrics
                best_epoch_metrics['threshold'] = best_threshold
                patience_counter = 0
                # Save only the model state_dict for efficiency
                torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pth'))
            else:
                patience_counter += 1

            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}',
                'Val PR_AUC': f'{pr_auc:.4f}', 'Best PR_AUC': f'{best_pr_auc:.4f}',
                'Patience': f'{patience_counter}/{args.patience}'
            })

            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        plot_loss_curves(train_losses, val_losses, fold, save_path=args.plot_dir)
        print(f"Fold {fold+1} Best Metrics (PR-AUC={best_pr_auc:.4f}):")
        if best_epoch_metrics:
            for key, val in best_epoch_metrics.items():
                print(f"  {key}: {val:.4f}")
            fold_metrics.append(best_epoch_metrics)

    print("\n===== K-Fold Cross-Validation Results =====")
    if fold_metrics:
        avg_metrics = {}
        for key in fold_metrics[0].keys():
            values = [m[key] for m in fold_metrics]
            avg = np.mean(values)
            std = np.std(values)
            avg_metrics[key] = (avg, std)
            print(f"Average {key.upper()}: {avg:.4f} ± {std:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train HierarchicalGNN for PPI with K-Fold Cross-Validation')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data pkl file')
    parser.add_argument('--model_dir', type=str, default='./saved_models', help='Directory to save models')
    parser.add_argument('--plot_dir', type=str, default='./plots', help='Directory to save loss plots')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of K-fold splits')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    # Model Hyperparameters
    parser.add_argument('--hidden_channels', type=int, default=256, help='Number of hidden channels in the GNN')
    parser.add_argument('--atom_num_layers', type=int, default=4, help='Number of layers in the atom-level GNN')
    parser.add_argument('--residue_num_layers', type=int, default=4, help='Number of layers in the residue-level GNN')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads in GATConv')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    # Training Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--scheduler_patience', type=int, default=10, help='Patience for learning rate scheduler')

    args = parser.parse_args()
    main(args)