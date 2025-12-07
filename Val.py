import torch
import os
from utils import calculate_metrics, find_best_threshold_by_f_beta, save_metrics_to_txt
from model import PPI
import config
from tqdm import tqdm
from Data import PPIData, PPIDataset, sparse_collate
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = config.DEVICE


def _load_model(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model = PPI(hid_dim=config.gcn_hid_dim, heads=config.HEADS, dropout=config.DROPOUT).to(device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    return model

@torch.no_grad()
def validate(model, val_loader, draw_plots=True, save_dir='./plots'):
    model.eval()
    all_prob, all_target = [], []
    for batch in tqdm(val_loader, desc='Validating'):
        batch = {
            k: v.to(config.DEVICE, non_blocking=True) if (torch.is_tensor(v) or hasattr(v, 'to')) else v
            for k, v in batch.items()
        }
        out = model(ax=batch['AA'], bx=batch['esm_c'], cx=batch['dssp'], dx=batch['BLOSUM'], ex=batch['pse'], fx=batch['res_atom'], adj=batch['adj'])
        probs = torch.sigmoid(out)
        all_prob.append(probs.detach().cpu())
        all_target.append(batch['y'].float().detach().cpu())
    all_targets_tensor = torch.cat(all_target, dim=0)
    all_probs_tensor = torch.cat(all_prob, dim=0)
    threshold, _ = find_best_threshold_by_f_beta(all_targets_tensor, all_probs_tensor, num_threshold=100)
    metrics = calculate_metrics(y_true=all_targets_tensor, y_scores=all_probs_tensor, threshold=threshold)
    if draw_plots:
        draw(all_targets_tensor.numpy(), all_probs_tensor.numpy(), save_dir=save_dir)
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

def validate_from_config(data_path, dset_name, model_path=None, draw_plots=True, save_dir='./plots'):
    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    data_loader = PPIData()
    all_proteins = data_loader.load_data(data_path)
    val_dataset = PPIDataset(all_proteins, sample_ratio=2, is_training=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=sparse_collate,
        pin_memory=True
    )
    print(f'Validation dataset size: {len(val_dataset)}')
    if model_path is None:
        model_path = os.path.join(config.PRE_MODEL, 'Model.pth')
    model = _load_model(model_path)
    print(f'Model loaded from {model_path}')
    metrics = validate(model, val_loader, save_dir=save_dir)
    print('\n' + '=' * 50)
    print(f'Evaluation Results for {dset_name}')
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
    save_metrics_to_txt(metrics, dset_name)
    print(f'✓ Metrics saved for {dset_name}')
    return metrics

def test_kfold_models(model_dir, model_fmt, test_data_path, k_folds=5, dset_name_prefix='Fold', save_dir='./plots'):
    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    data_loader = PPIData()
    all_proteins = data_loader.load_data(test_data_path)
    val_dataset = PPIDataset(all_proteins, sample_ratio=2, is_training=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=sparse_collate,
        pin_memory=True
    )
    print(f'Test dataset size: {len(val_dataset)}')
    metrics_list = []
    for fold in range(1, k_folds+1):
        model_path = os.path.join(model_dir, model_fmt.format(fold))
        print(f'Loading model for fold {fold}: {model_path}')
        model = _load_model(model_path)
        metrics = validate(model, val_loader, save_dir=os.path.join(save_dir, f'fold{fold}'))
        metrics_list.append(metrics)
        save_metrics_to_txt(metrics, f'./plots/{dset_name_prefix}{fold}')
        print(f'Fold {fold} metrics saved.')

    mean_metrics = {}

    keys = [k for k in metrics_list[0].keys() if isinstance(metrics_list[0][k], (int, float))]
    for key in keys:
        mean_metrics[key] = np.mean([m[key] for m in metrics_list])
    print('\n===== K-Fold Test Results =====')
    for i, m in enumerate(metrics_list, 1):
        print(f'Fold {i}:', {k: m[k] for k in keys})
    print('Mean:', mean_metrics)
    print('==============================\n')
    return metrics_list, mean_metrics

if __name__ == '__main__':
    model_dir = config.PRE_MODEL
    model_fmt = 'Model_fold{}.pth'
    k_folds = config.K_FOLDS
    test_data_path = config.VAL2
    dset_name_prefix = 'Fold'
    save_dir = './plots'
    print(f"Auto k-fold evaluation: model_dir={model_dir}, model_fmt={model_fmt}, k_folds={k_folds}, test_data_path={test_data_path}")
    test_kfold_models(
        model_dir=model_dir,
        model_fmt=model_fmt,
        test_data_path=test_data_path,
        k_folds=k_folds,
        dset_name_prefix=dset_name_prefix,
        save_dir=save_dir
    )
