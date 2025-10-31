from tqdm import tqdm
import torch
import os
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from utils import calculate_metrics, find_best_threshold_by_f_beta, plot_loss_curves, HybridLoss
from model import PPI
import config
from Data import Dataloader
device = config.DEVICE
print(torch.cuda.get_device_name(0))

def train(model, train_data, optimizer, data_loader):
    model.train()
    np.random.shuffle(train_data)

    total_loss = 0
    criterion = HybridLoss(
        pos_wt=torch.tensor(config.POS_WEIGHT, device=device),
        alpha=config.A,
        beta=config.B,
        device=device,
        ce_weight=config.B_WEIGHT,
        tversky_weight=config.T_WEIGHT
    )

    for i in range(0, len(train_data), config.BATCH_SIZE):
        batch_samples = train_data[i:i + config.BATCH_SIZE]
        optimizer.zero_grad()

        batch_loss_sum = 0
        for sample_data in batch_samples:
            data = data_loader.prepare_sample(sample_data)
            out = model(ax=data.b_a, bx=data.prot, cx=data.esm, adj=data.adj)
            loss = criterion.compute_loss(out, data.y)
            batch_loss_sum += loss.item()
            loss = loss / len(batch_samples)
            loss.backward()

        if config.GRAD_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_NORM)
        optimizer.step()

        total_loss += batch_loss_sum

    return total_loss / len(train_data)

@torch.no_grad()
def test(model, val_data, data_loader):
    model.eval()
    criterion = HybridLoss(
        pos_wt=torch.tensor(config.POS_WEIGHT, device=device),
        alpha=config.A,
        beta=config.B,
        device=device,
        ce_weight=config.B_WEIGHT,
        tversky_weight=config.T_WEIGHT
    )
    total_loss = 0
    all_probs, all_targets = [], []

    for sample_data in val_data:
        data = data_loader.prepare_sample(sample_data)
        out = model(ax=data.b_a, bx=data.prot, cx=data.esm, adj=data.adj)
        loss = criterion.compute_loss(out, data.y)
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

    model = PPI(in_channels=512, hid_dim=512, dropout=config.DROPOUT, lamda=config.LAMDA, alpha=config.ALPHA, training=True)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    warmup_epochs = 5

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda)
    reduce_lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=config.SCHEDULER_ETA_MIN)

    os.makedirs(config.PRE_MODEL, exist_ok=True)

    print("Starting training...")

    train_losses, val_losses = [], []
    best_loss = float('inf')
    patience_counter = 0

    epoch_pbar = tqdm(range(config.EPOCHS), desc="Training Progress", ncols=180)

    for epoch in epoch_pbar:
        train_loss = train(model, train_data, optimizer, data_loader)
        train_losses.append(train_loss)

        val_loss, metrics, best_threshold = test(model, val_data, data_loader)
        val_losses.append(val_loss)

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            reduce_lr_scheduler.step(val_loss)

        if val_losses[-1] < best_loss:
            best_loss = val_losses[-1]
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
    save_path=os.path.join(config.PLOT_DIR, f'Train.png')
    plot_loss_curves(train_losses, val_losses, save_path)

if __name__ == '__main__':
    main()