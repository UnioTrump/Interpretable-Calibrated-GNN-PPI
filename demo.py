from tqdm import tqdm
import torch
import os
import numpy as np
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from utils import calculate_metrics, find_best_threshold_by_f_beta, plot_loss_curves, HybridLoss
from model import PPI, SophiaG
import config
from Data import PPIData, PPIDataset, sparse_collate
from torch.utils.data import DataLoader

device = config.DEVICE
print(torch.cuda.get_device_name(0))


def train(model, train_loader, optimizer, loss_fun):
    model.train()
    total_loss = 0
    grad = []
    for idx, batch in enumerate(train_loader):
        batch = {
            k: v.to(config.DEVICE) if (torch.is_tensor(v) or hasattr(v, 'to')) else v
            for k, v in batch.items()
        }
        out = model(ax=batch['AA'], bx=batch['esm_c'], cx=batch['dssp'], dx=batch['BLOSUM'], adj=batch['adj'])
        loss = loss_fun(out, batch['y'])
        loss.backward()
        #=========detach gradient=========
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        grad.append(total_norm)
        #=================================
        optimizer.step(bs=config.BATCH_SIZE)
        if idx % 10 == 0:
            optimizer.update_hessian()
        optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item()
    # =========log grad=============
    avg_grad = sum(grad) / len(grad)
    max_grad = max(grad)
    if max_grad > 0:
        print(f'Average gradient norm: {avg_grad:.4f}\n'
              f'Max gradient norm: {max_grad:.4f}')
        # grad clip
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # ==============================

    return total_loss / len(train_loader)

@torch.no_grad()
def test(model, val_loader, loss_fun):
    model.eval()
    total_loss = 0
    all_probs, all_targets = [], []
    for batch in val_loader:
        batch = {
            k: v.to(config.DEVICE) if (torch.is_tensor(v) or hasattr(v, 'to')) else v
            for k, v in batch.items()
        }
        out = model(ax=batch['AA'], bx=batch['esm_c'], cx=batch['dssp'], dx=batch['BLOSUM'], adj=batch['adj'])
        loss = loss_fun(out, batch['y'])
        total_loss += loss.item()
        all_probs.append(torch.sigmoid(out))
        all_targets.append(batch['y'].float())
    avg_loss = total_loss / len(val_loader)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    all_probs_tensor = torch.cat(all_probs, dim=0)
    threshold, _ = find_best_threshold_by_f_beta(all_targets_tensor, all_probs_tensor, num_threshold=100)
    metrics = calculate_metrics(y_true=all_targets_tensor, y_scores=all_probs_tensor, threshold=threshold)
    return avg_loss, metrics, threshold


def main():

    all_proteins = PPIData.load_data(config.DATA_DIR)

    train_data, val_data = PPIData.split_data(all_proteins, train_ratio=0.8, seed=42)
    train_data = PPIDataset(train_data, sample_ratio=2, is_training=False)
    val_data = PPIDataset(val_data, sample_ratio=2, is_training=False)
    seed = config.SEED
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=sparse_collate)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, collate_fn=sparse_collate)

    print(f'Train_data: {len(train_data)}\nVal_data: {len(val_data)}')

    model = PPI(hid_dim=config.gcn_hid_dim, heads=config.HEADS, dropout=config.DROPOUT)
    model.to(device)
    print(f'参数量：{sum(p.numel() for p in model.parameters())}')

    optimizer = SophiaG(model.parameters(), lr=config.LEARNING_RATE, rho=0.05, weight_decay=config.WEIGHT_DECAY)
    warmup_epochs = 5

    def lr_lambda(EPOCH):
        if EPOCH < warmup_epochs:
            return (EPOCH + 1) / warmup_epochs
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda)
    reduce_lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    # reduce_lr_scheduler = CosineAnnealingLR(optimizer, T_max=config.T_MAX, eta_min=config.ETA_MIN)

    os.makedirs(config.PRE_MODEL, exist_ok=True)
    criterion = HybridLoss(
        alpha=config.A,
        beta=config.B,
        pos_wt=torch.tensor(0.3),     # Target: rise True prediction
        bce_weight=config.BCE_WEIGHT,
        focal_weight=config.FOCAL_WEIGHT,
        tversky_weight=config.Tversky_WEIGHT
    )
    train_losses, val_losses = [], []
    best_auprc = float('-inf')
    patience_counter = 0

    epoch_pbar = tqdm(range(config.EPOCHS), desc="Training Progress", ncols=180)

    for epoch in epoch_pbar:
        train_loss = train(model, train_loader, optimizer, criterion)
        train_losses.append(train_loss)

        val_loss, metrics, best_threshold = test(model, val_loader, criterion)
        val_losses.append(val_loss)

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            reduce_lr_scheduler.step(metrics['pr_auc'])

        if metrics['pr_auc'] > best_auprc:
            best_auprc = metrics['pr_auc']
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config.PRE_MODEL, f'Train.pth'))
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        epoch_pbar.set_postfix({
            "Train Loss": f"{train_loss:.4f}",
            "Val Loss": f"{val_loss:.4f}",
            "AUPRC": f"{metrics['pr_auc']:.4f}",
            "AUROC": f"{metrics['roc_auc']:.4f}",
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Patience": f"{patience_counter}/{config.PATIENCE}"
        })
    save_path = os.path.join(config.PLOT_DIR, f'Train.png')
    plot_loss_curves(train_losses, val_losses, save_path)


if __name__ == '__main__':
    main()