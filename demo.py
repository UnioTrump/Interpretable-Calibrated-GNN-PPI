from tqdm import tqdm
import torch
import os
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import WeightedCrossEntropy, calculate_metrics, find_best_threshold_by_f_beta
from GASPPI import DualStreamPPI
import matplotlib.pyplot as plt
import config
from data_utils import DataLoader

device = config.DEVICE
torch.manual_seed(config.SEED)

def train(model, train_proteins, optimizer, data_loader):
    model.train()
    np.random.shuffle(train_proteins)

    total_loss = 0
    criterion = WeightedCrossEntropy(pos_wt=torch.tensor(config.POS_WEIGHT, device=device), device=device)

    for i in range(0, len(train_proteins), config.BATCH_SIZE):
        batch_proteins = train_proteins[i:i + config.BATCH_SIZE]
        optimizer.zero_grad()

        batch_loss_sum = 0
        for protein in batch_proteins:
            data = data_loader.prepare_sample(protein)
            out = model(data)
            loss = criterion.compute_loss(out, data.y)
            batch_loss_sum += loss.item()
            loss = loss / len(batch_proteins)
            loss.backward()

        if config.GRAD_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_NORM)
        optimizer.step()

        total_loss += batch_loss_sum

    return total_loss / len(train_proteins)

@torch.no_grad()
def test(model, val_proteins, data_loader):
    model.eval()
    criterion = WeightedCrossEntropy(pos_wt=torch.tensor(config.POS_WEIGHT, device=device), device=device)
    total_loss = 0
    all_probs, all_targets = [], []

    for val_p in val_proteins:
        data = data_loader.prepare_sample(val_p)
        out = model(data)
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

def main():
    # 初始化数据加载器
    data_loader = DataLoader(device=device)

    # 加载数据
    all_proteins = data_loader.load_data(config.DATA_PATH)

    # 数据分割
    train_data, val_data = data_loader.split_data(all_proteins, train_ratio=0.8, seed=config.SEED)
    print(f'Training samples: {len(train_data)}')
    print(f'Validation samples: {len(val_data)}')

    # 获取数据信息
    data_info = data_loader.get_data_info(all_proteins[0])
    atom_in_channels = data_info['atom_in_channels']
    residue_in_channels = data_info['residue_in_channels']

    # Temporarily switch to the ablation model for diagnostics
    model_class = DualStreamPPI
    print(f"--- DIAGNOSTIC RUN: Using {model_class.__name__} ---")

    model = model_class(
        atom_in_channels=atom_in_channels,
        residue_in_channels=residue_in_channels,
        atom_hidden_dims=config.ATOM_HIDDEN_DIMS,
        residue_hidden_dims=config.RESIDUE_HIDDEN_DIMS,
        pe_dim=config.PE_DIM,
        geo_hidden_dims=config.GEO_HIDDEN_DIMS,
        fusion_hidden_dim=config.FUSION_HIDDEN_DIM,
        out_channels=config.OUT_CHANNELS,
        dropout=config.DROPOUT,
        heads=config.HEADS
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.SCHEDULER_T_MAX, eta_min=config.SCHEDULER_ETA_MIN)

    best_pr_auc = 0.0
    patience_counter = 0
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    best_model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    
    print("Starting training...")
    train_losses, val_losses = [], []
    epoch_pbar = tqdm(range(config.EPOCHS), desc="Training Progress", ncols=180)

    for epoch in epoch_pbar:
        train_loss = train(model, train_data, optimizer, data_loader)
        train_losses.append(train_loss)

        val_loss, metrics, best_threshold = test(model, val_data, data_loader)
        val_losses.append(val_loss)
        
        pr_auc = metrics['pr_auc']
        scheduler.step()

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}',
            'Val PR_AUC': f'{pr_auc:.4f}', 'Val ROC_AUC': f'{metrics["roc_auc"]:.4f}',
            'Patience': f'{patience_counter}/{config.PATIENCE}'
        })

        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    plot_save_path = os.path.join(config.PLOT_DIR, f'{config.PROJECT_NAME}_loss_curve.png')
    plot_loss_curves(train_losses, val_losses, save_path=plot_save_path)

if __name__ == '__main__':
    main()