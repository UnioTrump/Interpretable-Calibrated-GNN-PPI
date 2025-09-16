from tqdm import tqdm
import torch
import os
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import WeightedCrossEntropy, calculate_metrics, find_best_threshold_by_f_beta, plot_loss_curves
from GASPPI import DualStreamPPI
import config
from data_utils import DataLoader

device = config.DEVICE
print(torch.cuda.get_device_name(0))

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

def main():

    for index, seed in enumerate(config.SEED):
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        data_loader = DataLoader(device=device)
        all_proteins = data_loader.load_data(config.DATA_PATH)
        print('FUCKING Load Done!!!!!')
        train_data, val_data = data_loader.split_data(all_proteins, train_ratio=0.8, seed=42)
        print(f'Training samples: {len(train_data)}')
        print(f'Validation samples: {len(val_data)}')

        data_info = data_loader.get_data_info(all_proteins[0])
        atom_in_channels = data_info['atom_in_channels']
        residue_in_channels = data_info['residue_in_channels']

        model_class = DualStreamPPI
        model = model_class(
            atom_in_channels=atom_in_channels,
            residue_in_channels=residue_in_channels,
            pe_dim=config.PE_DIM,
            fusion_hidden_dim=config.FUSION_HIDDEN_DIM,
            out_channels=config.OUT_CHANNELS,
            dropout=config.DROPOUT,
            heads=config.HEADS
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        # scheduler = CosineAnnealingLR(optimizer, T_max=config.SCHEDULER_T_MAX, eta_min=config.SCHEDULER_ETA_MIN)
        os.makedirs(config.PRE_MODEL, exist_ok=True)

        print("Starting training...")
        print(f'Experiment {index}')
        train_losses, val_losses = [], []
        best_loss = 999
        patience_counter = 0

        epoch_pbar = tqdm(range(config.EPOCHS), desc="Training Progress", ncols=180)

        for epoch in epoch_pbar:
            train_loss = train(model, train_data, optimizer, data_loader)
            train_losses.append(train_loss)

            val_loss, metrics, best_threshold = test(model, val_data, data_loader)
            val_losses.append(val_loss)

            pr_auc = metrics['pr_auc']
            # scheduler.step()

            if val_losses[-1] < best_loss:
                best_loss = val_losses[-1]
                patience_counter = 0
                best_model_path = os.path.join(config.PRE_MODEL, f'{index}_best_model.pth')
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

        plot_save_path = os.path.join(config.PLOT_DIR, f'{index}_loss_curve.png')
        plot_loss_curves(train_losses, val_losses, save_path=plot_save_path)

if __name__ == '__main__':
    main()