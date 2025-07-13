import os
import torch
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

def find_best_threshold_by_mcc(y_true, y_scores, num_thresholds=100):
    """通过MCC值寻找最佳阈值"""

    y_true = y_true.cpu().numpy()
    y_scores = y_scores.cpu().numpy()
    thresholds = np.linspace(0, 1, num_thresholds)
    mcc_values = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)
        mcc_values.append(mcc)

    best_idx = np.argmax(mcc_values)
    best_threshold = thresholds[best_idx]
    best_mcc = mcc_values[best_idx]

    return best_threshold, best_mcc

def find_best_threshold_by_f_beta(y_true, y_scores, num_threshold, beta=1.5):
    """
        通过Fβ寻找最佳阈值

        Args:
            beta: precision和recall的权重值
    """

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

def calculate_metrics(y_true, y_scores, threshold):
    """计算评估指标"""
    y_true = y_true.cpu().numpy()
    y_scores = y_scores.cpu().numpy()
    y_pred = (y_scores >= threshold).astype(int)

    # 计算基础指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    # 计算混淆矩阵元素
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # 计算AUC指标
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall_curve, precision_curve)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mcc': mcc,
        'specificity': specificity,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }