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
import argparse
import glob

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
        out = model(ax=batch['AA'], bx=batch['esm_c'], cx=batch['dssp'], dx=batch['BLOSUM'], ex=batch['pse'],
                    fx=batch['res_atom'], adj=batch['adj'])
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
        out = model(ax=batch['AA'], bx=batch['esm_c'], cx=batch['dssp'], dx=batch['BLOSUM'], ex=batch.get('pse'), fx=batch.get('res_atom'), adj=batch['adj'])
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
    parser = argparse.ArgumentParser(description='Fit temperature for trained PPI model')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained model (pth file)')
    parser.add_argument('--model_dir', type=str, default=None, help='Directory containing fold model files')
    parser.add_argument('--data_path', type=str, default=None, help='Path to calibration data folder (no train/val re-split)')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save calibrated model(s)')
    args = parser.parse_args()

    # calibration / tuning set: should be pre-split and not overlap with training data
    data_path = args.data_path if args.data_path else config.VAL3
    all_proteins = PPIData.load_data(data_path)

    calib_data = PPIDataset(all_proteins, sample_ratio=2, is_training=False)

    seed = config.SEED
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    val_loader = DataLoader(calib_data, batch_size=1, shuffle=False, collate_fn=sparse_collate)
    print(f'Calibration data size: {len(calib_data)}')
    criterion = HybridLoss(
        alpha=config.A,
        beta=config.B,
        pos_wt=torch.tensor(0.1),
        bce_weight=config.BCE_WEIGHT,
        focal_weight=config.FOCAL_WEIGHT,
        tversky_weight=config.Tversky_WEIGHT
    )

    # 处理多个fold模型
    model_files = []
    if args.model_dir:
        model_files = sorted(glob.glob(os.path.join(args.model_dir, 'Model_fold*.pth')))
        if not model_files:
            print(f'No fold model files found in {args.model_dir}')
            return
    elif args.model_path:
        model_files = [args.model_path]
    else:
        model_files = [os.path.join(config.PRE_MODEL, 'Model.pth')]

    for model_path in model_files:
        fold_name = os.path.splitext(os.path.basename(model_path))[0]
        model = PPI(hid_dim=config.gcn_hid_dim, heads=config.HEADS, dropout=config.DROPOUT)
        model.to(device)
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print(f'Loaded model from {model_path}')
        model.eval()
        _, _, _, logits_tensor, targets_tensor = test(model, val_loader, criterion)
        T = fit_T(logits_tensor, targets_tensor.view(-1, 1))
        loss, metrics, threshold = test_T(model, val_loader, criterion, T)
        print(f'Temperature T: {T.item():.4f}')
        if args.save_path:
            save_path = os.path.join(args.save_path, f'{fold_name}_calibrated.pth')
        elif args.model_dir:
            save_path = os.path.join(args.model_dir, f'{fold_name}_calibrated.pth')
        else:
            save_path = os.path.join(config.PRE_MODEL, f'{fold_name}_calibrated.pth')
        torch.save({
            'model': model.state_dict(),
            'T': T.cpu()
        }, save_path)
        print(f'Calibrated model saved to {save_path}')
        print('=' * 50)
        print(f'Results for {fold_name}:')
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