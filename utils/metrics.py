import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef


def calculate_metrics(y_true, y_scores, threshold):
    """Calculate and return a dictionary of binary classification metrics."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()

    pr_auc = average_precision_score(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    y_pred = (y_scores >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'mcc': mcc,
        'threshold': threshold,
        'confusion_matrix': {
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }
    }

    return metrics


def find_best_threshold_by_mcc(y_true, y_scores, num_thresholds=100):
    """
    通过MCC值寻找最佳阈值

    参数:
    - y_true: 真实标签
    - y_scores: 预测为正类的概率
    - num_thresholds: 要尝试的阈值数量

    返回:
    - best_threshold: 最佳阈值 (MCC值最高)
    - best_mcc: 对应的最佳MCC值
    - mcc_values: 所有阈值对应的MCC值列表
    - thresholds: 尝试的阈值列表
    """
    # 生成阈值范围
    thresholds = np.linspace(0, 1, num_thresholds)
    mcc_values = []

    # 计算每个阈值下的MCC值
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)
        mcc_values.append(mcc)

    # 找到最佳阈值和MCC值
    best_idx = np.argmax(mcc_values)
    best_threshold = thresholds[best_idx]
    best_mcc = mcc_values[best_idx]

    return best_threshold, best_mcc, mcc_values, thresholds