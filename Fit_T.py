from torch import nn
import torch
import os
import numpy as np
from utils import calculate_metrics, find_best_threshold_by_f_beta, HybridLoss
from model import PPI
import config
from Data import PPIData, PPIDataset, sparse_collate
from torch.utils.data import DataLoader
from tqdm import tqdm

device = config.DEVICE

@torch.no_grad()
def test(model, val_loader, loss_fun):
    model.eval()
    total_loss = 0
    all_logits, all_probs, all_targets = [], [], []
    for batch in tqdm(val_loader, total=len(val_loader)):
        batch = {
            k: v.to(config.DEVICE) if (torch.is_tensor(v) or hasattr(v, 'to')) else v
            for k, v in batch.items()
        }
        out = model(ax=batch['AA'], bx=batch['esm_c'], cx=batch['dssp'], dx=batch['BLOSUM'], adj=batch['adj'])
        loss = loss_fun(out, batch['y'])
        total_loss += loss.item()
        all_logits.append(out)
        all_probs.append(torch.sigmoid(out))
        all_targets.append(batch['y'].float())
    avg_loss = total_loss / len(val_loader)
    logits_tensor = torch.cat(all_logits, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    probs_tensor = torch.cat(all_probs, dim=0)
    threshold, _ = find_best_threshold_by_f_beta(targets_tensor, probs_tensor, num_threshold=100)
    metrics = calculate_metrics(y_true=targets_tensor, y_scores=probs_tensor, threshold=threshold)
    return avg_loss, metrics, threshold, logits_tensor, targets_tensor


@torch.no_grad()
def test_T(model, val_loader, loss_fun, T):
    model.eval()
    total_loss = 0
    all_probs, all_targets = [], []
    for batch in val_loader:
        batch = {
            k: v.to(config.DEVICE) if (torch.is_tensor(v) or hasattr(v, 'to')) else v
            for k, v in batch.items()
        }
        out = model(ax=batch['AA'], bx=batch['esm_c'], cx=batch['dssp'], dx=batch['BLOSUM'], adj=batch['adj'])
        out = out / T
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

class Temp(nn.Module):
    def __init__(self):
        super(Temp, self).__init__()
        self.T = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x / self.T

def fit_T(logits, labels):
    scaler = Temp().to(device)
    optimizer_T = torch.optim.LBFGS([scaler.T], lr=config.LEARNING_RATE, max_iter=50)
    def closure():
        optimizer_T.zero_grad()
        scaled_logits = scaler(logits)
        loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.1))(scaled_logits, labels.float())
        loss.backward()
        return loss.detach()
    optimizer_T.step(closure)
    return scaler.T.detach().clone()

def main():

    all_proteins = PPIData.load_data(config.DATA_DIR)

    _, val_data = PPIData.split_data(all_proteins, train_ratio=0.8, seed=42)
    # train_data = PPIDataset(train_data, sample_ratio=2, is_training=False)
    val_data = PPIDataset(val_data, sample_ratio=2, is_training=False)
    seed = config.SEED
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, collate_fn=sparse_collate)

    print(f'Val_data: {len(val_data)}')

    criterion = HybridLoss(
        alpha=config.A,
        beta=config.B,
        pos_wt=torch.tensor(0.1),     # Target: rise True prediction
        bce_weight=config.BCE_WEIGHT,
        focal_weight=config.FOCAL_WEIGHT,
        tversky_weight=config.Tversky_WEIGHT
    )

    model = PPI(hid_dim=config.gcn_hid_dim, heads=config.HEADS, dropout=config.DROPOUT)
    model.to(device)
    # Temperature Scaling
    model.load_state_dict(torch.load(os.path.join(config.PRE_MODEL, f'Train.pth')))
    print('Load Successfully!')
    model.eval()
    _, _, _, logits_tensor, targets_tensor = test(model, val_loader, criterion)
    T = fit_T(logits_tensor, targets_tensor.view(-1, 1))
    loss, metrics, threshold = test_T(model, val_loader, criterion, T)
    print(f'Temperature T: {T.item():.4f}')

    save_path = os.path.join(config.PRE_MODEL, f'Model.pth')
    torch.save({
        'model': model.state_dict(),
        'T': T.cpu()
    }, save_path)

    print('=' * 50)
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            print(f"  {key.replace('_', ' ').title()}: {val:.4f}")
        elif isinstance(val, dict):
            print(f"  {key.replace('_', ' ').title()}:")
            for sub_key, sub_val in val.items():
                print(f"    {sub_key.upper()}: {sub_val}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {val}")
    print('=' * 50 + '\n')

if __name__ == '__main__':
    main()