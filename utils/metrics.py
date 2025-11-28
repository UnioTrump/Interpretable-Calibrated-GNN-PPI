import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
)
import config

def calculate_metrics(y_true, y_scores, threshold):
    """
    计算二分类常用指标。
    Args:
        y_true: 真实标签 (array-like or torch.Tensor)
        y_scores: 预测概率 (array-like or torch.Tensor)
        threshold: 二分类阈值
    """
    y_true = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else np.asarray(y_true)
    y_scores = y_scores.cpu().numpy() if isinstance(y_scores, torch.Tensor) else np.asarray(y_scores)

    pr_auc = average_precision_score(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    y_pred = (y_scores >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    # confusion_matrix 健壮处理
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'mcc': mcc,
        'threshold': threshold,
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    }


def find_best_threshold_by_mcc(y_true, y_scores, num_thresholds=100):
    """
    通过MCC值寻找最佳阈值。
    Args:
        y_true: 真实标签
        y_scores: 预测概率
        num_thresholds: 尝试的阈值数量
    """
    y_true = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else np.asarray(y_true)
    y_scores = y_scores.cpu().numpy() if isinstance(y_scores, torch.Tensor) else np.asarray(y_scores)
    thresholds = np.linspace(0, 1, num_thresholds)
    mcc_values = [matthews_corrcoef(y_true, (y_scores >= t).astype(int)) for t in thresholds]
    best_idx = np.argmax(mcc_values)
    return thresholds[best_idx], mcc_values[best_idx], mcc_values, thresholds

def find_best_threshold_by_f_beta(y_true, y_scores, num_threshold, beta=config.F_BETA):
    """Find the best threshold by maximizing the F-beta score."""

    thresholds = np.linspace(0, 1, num_threshold)
    y_true = y_true.cpu().numpy()
    y_scores = y_scores.cpu().numpy()
    f_beta_values = []
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        precisions = precision_score(y_true, y_pred, zero_division=0)
        recalls = recall_score(y_true, y_pred)
        f_beta = (1 + beta**2)*(precisions * recalls)/(beta**2*precisions+recalls) if (precisions + recalls) != 0 else 0
        f_beta_values.append(f_beta)

    best_idx = np.argmax(f_beta_values)
    best_threshold = thresholds[best_idx]
    best_f_beta = f_beta_values[best_idx]
    return best_threshold, best_f_beta


def find_best_threshold_by_accuracy(y_true, y_scores, num_thresholds=200, refine=True):
    """新增: 通过Accuracy寻找最佳阈值, 并在初次找到后做局部细化搜索"""
    y_true = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else np.asarray(y_true)
    y_scores = y_scores.cpu().numpy() if isinstance(y_scores, torch.Tensor) else np.asarray(y_scores)
    thresholds = np.linspace(0, 1, num_thresholds)
    accs = []
    for t in thresholds:
        accs.append(accuracy_score(y_true, (y_scores >= t).astype(int)))
    best_idx = int(np.argmax(accs))
    best_t = float(thresholds[best_idx])  # 转换为Python float 解决类型警告
    best_acc = float(accs[best_idx])
    if refine:
        low = max(0.0, best_t - 0.05)
        high = min(1.0, best_t + 0.05)
        fine_ts = np.linspace(low, high, num_thresholds)
        fine_accs = []
        for t in fine_ts:
            fine_accs.append(accuracy_score(y_true, (y_scores >= t).astype(int)))
        f_best_idx = int(np.argmax(fine_accs))
        best_t = float(fine_ts[f_best_idx])
        best_acc = float(fine_accs[f_best_idx])
    return best_t, best_acc