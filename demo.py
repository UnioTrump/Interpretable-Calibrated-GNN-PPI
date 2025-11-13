from tqdm import tqdm
import torch
import os
import numpy as np
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from utils import calculate_metrics, find_best_threshold_by_f_beta, plot_loss_curves, HybridLoss
from model import PPI, SAM
import config
from Data import Dataloader

device = config.DEVICE
print(torch.cuda.get_device_name(0))


def train(model, train_data, optimizer, data_loader, loss_fn):
    model.train()
    np.random.shuffle(train_data)

    total_loss = 0
    criterion = loss_fn

    for i in range(0, len(train_data), config.BATCH_SIZE):
        batch_samples = train_data[i:i + config.BATCH_SIZE]

        batch_loss_sum = 0
        for sample_data in batch_samples:
            data = data_loader.prepare_sample(sample_data)
            out = model(ax=data.aa, bx=data.esm, cx=data.prot, adj=data.adj)
            loss = criterion.forward(out, data.y)
            batch_loss_sum += loss.item()
            loss = loss / len(batch_samples)

            def closure():
                l = criterion(model(ax=data.aa, bx=data.esm, cx=data.prot, adj=data.adj), data.y)
                l.backward()
                return l
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.15)
            optimizer.step(closure)

        total_loss += batch_loss_sum

    return total_loss / len(train_data)


@torch.no_grad()
def test(model, val_data, data_loader, loss_fun):
    model.eval()
    criterion = loss_fun
    total_loss = 0
    all_probs, all_targets = [], []

    for sample_data in val_data:
        data = data_loader.prepare_sample(sample_data)
        out = model(ax=data.aa, bx=data.esm, cx=data.prot, adj=data.adj)
        loss = criterion.forward(out, data.y)
        total_loss += loss.item()
        all_probs.append(torch.sigmoid(out))
        all_targets.append(data.y.float())

    avg_loss = total_loss / len(val_data)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    all_probs_tensor = torch.cat(all_probs, dim=0)

    threshold, _ = find_best_threshold_by_f_beta(all_targets_tensor, all_probs_tensor, num_threshold=100)
    metrics = calculate_metrics(y_true=all_targets_tensor, y_scores=all_probs_tensor, threshold=threshold)

    return avg_loss, metrics, threshold


def main():
    data_loader = Dataloader()
    all_proteins = Dataloader.load_data(config.DATA_DIR)
    train_data, val_data = Dataloader.split_data(all_proteins, train_ratio=0.8, seed=42)

    seed = config.SEED
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = PPI(hid_dim=config.gcn_hid_dim, heads=config.HEADS, dropout=config.DROPOUT)
    model.to(device)
    print(f'参数量：{sum(p.numel() for p in model.parameters())}')

    base_optimizer = torch.optim.Adam
    optimizer = SAM(model.parameters(), base_optimizer=base_optimizer, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    warmup_epochs = 5

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda)
    reduce_lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    # reduce_lr_scheduler = CosineAnnealingLR(optimizer, T_max=config.T_MAX, eta_min=config.ETA_MIN)

    os.makedirs(config.PRE_MODEL, exist_ok=True)
    criterion = HybridLoss(
        alpha=config.A,
        beta=config.B,
        pos_wt=torch.tensor(0.3),
        bce_weight=config.BCE_WEIGHT,
        focal_weight=config.FOCAL_WEIGHT,
        tversky_weight=config.Tversky_WEIGHT
    )
    train_losses, val_losses = [], []
    best_auprc = float('-inf')
    patience_counter = 0

    epoch_pbar = tqdm(range(config.EPOCHS), desc="Training Progress", ncols=180)

    for epoch in epoch_pbar:
        train_loss = train(model, train_data, optimizer, data_loader, criterion)
        train_losses.append(train_loss)

        val_loss, metrics, best_threshold = test(model, val_data, data_loader, criterion)
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
            "Patience": f"{patience_counter}/{config.PATIENCE}"
        })
    save_path = os.path.join(config.PLOT_DIR, f'Train.png')
    plot_loss_curves(train_losses, val_losses, save_path)


if __name__ == '__main__':
    main()
