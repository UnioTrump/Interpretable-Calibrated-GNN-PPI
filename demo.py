from tqdm import tqdm
import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from utils import calculate_metrics, find_best_threshold_by_f_beta, plot_loss_curves, HybridLoss
from model import PPI, SophiaG
import config
from Data import PPIData, PPIDataset, sparse_collate
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from Fit_T import fit_T, test_T, getlogits
import os

device = config.DEVICE
print(torch.cuda.get_device_name(0))


def train(model, train_loader, optimizer, loss_fun):
    model.train()
    total_loss = 0
    # grad = []
    for idx, batch in enumerate(train_loader):
        batch = {
            k: v.to(config.DEVICE) if (torch.is_tensor(v) or hasattr(v, 'to')) else v
            for k, v in batch.items()
        }
        out = model(ax=batch['AA'], bx=batch['esm_c'], cx=batch['dssp'], dx=batch['BLOSUM'],ex=batch['pse'],fx=batch['res_atom'], adj=batch['adj'])
        loss = loss_fun(out, batch['y'])
        loss.backward()
        #=========detach gradient=========
        '''
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        grad.append(total_norm)
        '''
        #=================================
        optimizer.step(bs=config.BATCH_SIZE)
        if idx % 10 == 0:
            optimizer.update_hessian()
        optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item()
    # =========log grad=============
    '''
    avg_grad = sum(grad) / len(grad)
    max_grad = max(grad)
    if max_grad > 0:
        print(f'Average gradient norm: {avg_grad:.4f}\n'
              f'Max gradient norm: {max_grad:.4f}')
        # grad clip
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    '''
    # ==============================

    return total_loss / len(train_loader)

@torch.no_grad()
def validata(model, val_loader, loss_fun):
    model.eval()
    total_loss = 0
    all_probs, all_targets = [], []
    for batch in val_loader:
        batch = {
            k: v.to(config.DEVICE) if (torch.is_tensor(v) or hasattr(v, 'to')) else v
            for k, v in batch.items()
        }
        out = model(ax=batch['AA'], bx=batch['esm_c'], cx=batch['dssp'], dx=batch['BLOSUM'],ex=batch['pse'],fx=batch['res_atom'], adj=batch['adj'])
        loss = loss_fun(out, batch['y'])
        total_loss += loss.item()
        all_probs.append(torch.sigmoid(out))
        all_targets.append(batch['y'].float())
    avg_loss = total_loss / len(val_loader)
    targets_tensor = torch.cat(all_targets, dim=0)
    probs_tensor = torch.cat(all_probs, dim=0)
    threshold = find_best_threshold_by_f_beta(targets_tensor, probs_tensor, num_threshold=100)
    metrics = calculate_metrics(y_true=targets_tensor, y_scores=probs_tensor, threshold=threshold)
    return avg_loss, metrics, threshold

@torch.no_grad()
def test(model, test_loader, loss_fun, T, threshold):
    model.eval()
    total_loss = 0
    all_probs, all_targets = [], []
    for batch in test_loader:
        batch = {
            k: v.to(config.DEVICE) if (torch.is_tensor(v) or hasattr(v, 'to')) else v
            for k, v in batch.items()
        }
        out = model(ax=batch['AA'], bx=batch['esm_c'], cx=batch['dssp'], dx=batch['BLOSUM'],ex=batch['pse'],fx=batch['res_atom'], adj=batch['adj'])
        if T is not None:
            out = out / T
        loss = loss_fun(out, batch['y'])
        total_loss += loss.item()
        all_probs.append(torch.sigmoid(out))
        all_targets.append(batch['y'].float())
    avg_loss = total_loss / len(test_loader)
    targets = torch.cat(all_targets, dim=0)
    probs = torch.cat(all_probs, dim=0)
    metrics = calculate_metrics(y_true=targets, y_scores=probs, threshold=threshold)
    return avg_loss, metrics

def cross_validate():
    """
        The dataset is divided into an 8:1:1 ratio,
        serving as the training set, calibration set, and test set, respectively.
        To prevent underfitting caused by a small dataset,
        K-Fold cross-validation was performed on the training set.
    """
    all_proteins = PPIData.load_data(config.DATA_DIR)

    train_all, test_all, calib_all = PPIData.split_data(
        all_proteins,
        train_ratio=0.8,
        test_ratio=0.1,
        seed=config.SEED,
    )

    calib_dataset = PPIDataset(calib_all)
    test_dataset = PPIDataset(test_all)

    calib_loader = DataLoader(
        calib_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=sparse_collate
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=sparse_collate
    )

    print(f"Total samples: {len(all_proteins)} | Train: {len(train_all)} | Test: {len(test_all)} | Calib: {len(calib_all)}")


    kf = KFold(n_splits=config.K_FOLDS, shuffle=True, random_state=config.SEED)
    auprc_scores = []
    train_array = np.array(train_all)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_array)):
        fold_train_data = train_array[train_idx].tolist()
        fold_val_data = train_array[val_idx].tolist()

        fold_train_data = PPIDataset(fold_train_data)
        fold_val_data = PPIDataset(fold_val_data)

        torch.cuda.manual_seed_all(config.SEED)
        np.random.seed(config.SEED)
        torch.manual_seed(config.SEED)

        train_loader = DataLoader(fold_train_data, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=sparse_collate)
        val_loader = DataLoader(fold_val_data, batch_size=1, shuffle=False, collate_fn=sparse_collate)

        model = PPI(hid_dim=config.gcn_hid_dim, heads=config.HEADS, dropout=config.DROPOUT)
        model.to(device)

        optimizer = SophiaG(model.parameters(), lr=config.LEARNING_RATE, rho=0.05, weight_decay=config.WEIGHT_DECAY)

        warmup_epochs = 5

        def lr_lambda(EPOCH):
            if EPOCH < warmup_epochs:
                return (EPOCH + 1) / warmup_epochs
            return 1.0

        warmup_scheduler = LambdaLR(optimizer, lr_lambda)
        reduce_lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        criterion = HybridLoss(
            alpha=config.A,
            beta=config.B,
            pos_wt=torch.tensor(0.1),
            bce_weight=config.BCE_WEIGHT,
            tversky_weight=config.Tversky_WEIGHT,
            focal_weight=config.FOCAL_WEIGHT,
        )

        best_auprc = float('-inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        save_dir = config.PRE_MODEL
        os.makedirs(save_dir, exist_ok=True)

        epoch_iter = tqdm(range(config.EPOCHS), desc=f"Fold {fold+1}", ncols=180)
        for epoch in epoch_iter:
            train_loss = train(model, train_loader, optimizer, criterion)
            val_loss, metrics, _ = validata(model, val_loader, criterion)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                reduce_lr_scheduler.step(metrics['pr_auc'])

            if metrics['pr_auc'] > best_auprc:
                best_auprc = metrics['pr_auc']
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(save_dir, f'Model_fold{fold+1}.pth'))
            else:
                patience_counter += 1

            if patience_counter >= config.PATIENCE:
                break

            epoch_iter.set_postfix({
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_auprc": f"{metrics['pr_auc']:.4f}",
                "val_auroc": f"{metrics['roc_auc']:.4f}",
                "acc": f"{metrics['accuracy']:.4f}",
                "patience": f"{patience_counter}",
            })

        plot_loss_curves(train_losses, val_losses, save_path=f"{config.PLOT_DIR}/loss_curve_fold{fold+1}.png")
        print(f"Fold {fold+1}: Best AUPRC = {best_auprc:.4f}")
        auprc_scores.append(best_auprc)

        # === Temperature scaling immediately after training this fold ===
        best_model_path = os.path.join(save_dir, f'Model_fold{fold+1}.pth')
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        # ===== Calibration =====
        logits, targets = getlogits(model, calib_loader)

        T = fit_T(logits, targets.view(-1, 1))
        print(f"[Fold {fold + 1}] Fitted temperature T = {T.item():.4f}")

        threshold = test_T(logits, targets, T)
        cal_metrics = calculate_metrics(logits, targets, threshold)

        print(
            f"[Fold {fold + 1}] Calibration: "
            f"F1={cal_metrics['f1']:.4f}, "
            f"Threshold={threshold:.4f}"
        )

        # === Test ===
        test_loss, test_metrics = test(model, test_loader, criterion, T, threshold)

        print(
            f"[Fold {fold + 1}], "
            f"ACC={test_metrics['accuracy']:.4f}, "
            f"Test AUPRC={test_metrics['pr_auc']:.4f}, "
            f"AUROC={test_metrics['roc_auc']:.4f}, "
            f"F1={test_metrics['f1']:.4f}"
        )

        torch.save({
            'model': model.state_dict(),
            'T': T.cpu(),
            'threshold': threshold.cpu()
        }, os.path.join(save_dir, f'Model_fold{fold+1}_calibrated.pth'))

if __name__ == '__main__':
    # main()
    cross_validate()
