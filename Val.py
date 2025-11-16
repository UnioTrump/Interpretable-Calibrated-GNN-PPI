import torch
import os
from utils import calculate_metrics, find_best_threshold_by_f_beta, save_metrics_to_txt
from model import PPI
import config
from tqdm import tqdm
from Data import PPIData, PPIDataset, sparse_collate
import numpy as np
import argparse
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = config.DEVICE


@torch.no_grad()
def E(model, val_loader):

    model.eval()
    all_prob, all_target = [], []

    for batch in tqdm(val_loader, desc='Testing'):

        batch = {
            k: v.to(config.DEVICE, non_blocking=True) if (torch.is_tensor(v) or hasattr(v, 'to')) else v
            for k, v in batch.items()
        }

        out = model(ax=batch['AA'], bx=batch['esm_c'], cx=batch['prot'], adj=batch['adj'])

        probs = torch.sigmoid(out)
        all_prob.append(probs.detach().cpu())
        all_target.append(batch['y'].float().detach().cpu())

    all_targets_tensor = torch.cat(all_target, dim=0)
    all_probs_tensor = torch.cat(all_prob, dim=0)

    threshold, _ = find_best_threshold_by_f_beta(all_targets_tensor, all_probs_tensor, num_threshold=100)
    metrics = calculate_metrics(y_true=all_targets_tensor, y_scores=all_probs_tensor, threshold=threshold)
    draw(all_targets_tensor.numpy(), all_probs_tensor.numpy())

    return metrics


def draw(y_true, y_pred, save_dir='./plots'):

    os.makedirs(save_dir, exist_ok=True)

    # ========== ROC 曲线 ==========
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ========== PR 曲线 ==========
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    aupr = average_precision_score(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUPR = {aupr:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Plots saved to {save_dir}/")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='PPI Validation')
    parser.add_argument('--data', required=True, type=str, help='Data path variable name')
    parser.add_argument('--Dset_name', required=True, type=str, help='Dataset name for saving results')
    args = parser.parse_args()

    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    data_loader = PPIData()
    all_proteins = data_loader.load_data(eval(args.data))

    val_dataset = PPIDataset(all_proteins, sample_ratio=2, is_training=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE // 2,
        shuffle=False,
        collate_fn=sparse_collate,
        pin_memory=True
    )

    print(f'Validation dataset size: {len(val_dataset)}')

    model = PPI(hid_dim=config.gcn_hid_dim, heads=config.HEADS, dropout=config.DROPOUT).to(device)
    best_model_path = os.path.join(config.PRE_MODEL, 'Train.pth')

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model file not found: {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    print(f'✓ Model loaded from {best_model_path}')

    metrics = E(model, val_loader)

    print('\n' + '=' * 50)
    print(f'Evaluation Results for {args.Dset_name}')
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

    save_metrics_to_txt(metrics, args.Dset_name)
    print(f'✓ Metrics saved for {args.Dset_name}')

if __name__ == '__main__':
    main()