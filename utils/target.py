import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef


def calculate_metrics(y_true, y_pred, y_scores=None, threshold=0.5):
    """
    计算常见的二分类评估指标

    参数:
    - y_true: 真实标签 (0 或 1)
    - y_pred: 预测标签 (0 或 1)
    - y_scores: 预测为正类的概率 (用于ROC和PR曲线)
    - threshold: 二分类的阈值

    返回:
    - 包含各指标值的字典
    """
    # 确保输入是numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if y_scores is not None and isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()

    # 计算混淆矩阵元素
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 计算基础指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # 也称为灵敏度(Sensitivity)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特异度
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)  # Matthews相关系数

    # 计算其他指标
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # 负预测值
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # 假正率
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # 假负率

    # 创建指标字典
    metrics = {
        'accuracy': accuracy,
        'precision': precision,  # 正预测值 (Positive Predictive Value)
        'recall': recall,  # 敏感度 (Sensitivity)
        'specificity': specificity,
        'f1_score': f1,
        'mcc': mcc,
        'npv': npv,  # 负预测值 (Negative Predictive Value)
        'fpr': fpr,  # 假正率 (False Positive Rate)
        'fnr': fnr,  # 假负率 (False Negative Rate)
        'confusion_matrix': {
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        }
    }

    # 计算ROC和PR曲线相关指标 (如果提供了分数)
    if y_scores is not None:
        # ROC曲线和AUC
        fpr_curve, tpr_curve, thresholds_roc = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr_curve, tpr_curve)

        # PR曲线和AUC
        precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall_curve, precision_curve)

        metrics.update({
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'roc_curve': {
                'fpr': fpr_curve,
                'tpr': tpr_curve,
                'thresholds': thresholds_roc
            },
            'pr_curve': {
                'precision': precision_curve,
                'recall': recall_curve,
                'thresholds': thresholds_pr
            }
        })

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


def plot_mcc_thresholds(thresholds, mcc_values, best_threshold, best_mcc, title='MCC vs. Threshold'):
    """
    绘制MCC值随阈值变化的曲线

    参数:
    - thresholds: 阈值列表
    - mcc_values: MCC值列表
    - best_threshold: 最佳阈值
    - best_mcc: 最佳MCC值
    - title: 图表标题
    """
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, mcc_values, 'b-', label='MCC')
    plt.axvline(x=best_threshold, color='r', linestyle='--',
                label=f'Best Threshold: {best_threshold:.3f}\nMCC: {best_mcc:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('MCC')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图像
    plt.savefig('mcc_vs_threshold.png', dpi=300, bbox_inches='tight')
    plt.show()
    return plt.gcf()


def plot_confusion_matrix(y_true, y_pred, labels=None, title='Confusion Matrix'):
    """
    绘制混淆矩阵

    参数:
    - y_true: 真实标签
    - y_pred: 预测标签
    - labels: 类别标签
    - title: 图表标题
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(y_true, y_scores, title='ROC Curve'):
    """
    绘制ROC曲线

    参数:
    - y_true: 真实标签
    - y_scores: 预测为正类的概率
    - title: 图表标题
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_pr_curve(y_true, y_scores, title='Precision-Recall Curve'):
    """
    绘制精确率-召回率(PR)曲线

    参数:
    - y_true: 真实标签
    - y_scores: 预测为正类的概率
    - title: 图表标题
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.axhline(y=sum(y_true) / len(y_true), color='navy', linestyle='--',
                label=f'Baseline ({sum(y_true) / len(y_true):.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('pr_curve.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_metrics_report(metrics):
    """
    打印评估指标报告

    参数:
    - metrics: 由calculate_metrics函数返回的指标字典
    """
    print('=' * 60)
    print('二分类评估指标报告'.center(50))
    print('=' * 60)

    # 混淆矩阵
    cm = metrics['confusion_matrix']
    print(f"混淆矩阵:")
    print(f"真阳性(TP): {cm['tp']}, 假阳性(FP): {cm['fp']}")
    print(f"假阴性(FN): {cm['fn']}, 真阴性(TN): {cm['tn']}")
    print('-' * 60)

    # 基本指标
    print(f"准确率(Accuracy): {metrics['accuracy']:.4f}")
    print(f"精确率(Precision): {metrics['precision']:.4f}")
    print(f"召回率(Recall/Sensitivity): {metrics['recall']:.4f}")
    print(f"特异度(Specificity): {metrics['specificity']:.4f}")
    print(f"F1分数: {metrics['f1_score']:.4f}")
    print(f"Matthews相关系数(MCC): {metrics['mcc']:.4f}")
    print('-' * 60)

    # 额外指标
    print(f"负预测值(NPV): {metrics['npv']:.4f}")
    print(f"假正率(FPR): {metrics['fpr']:.4f}")
    print(f"假负率(FNR): {metrics['fnr']:.4f}")
    print('-' * 60)

    # AUC指标
    if 'roc_auc' in metrics:
        print(f"ROC曲线下面积(AUC): {metrics['roc_auc']:.4f}")
        print(f"PR曲线下面积: {metrics['pr_auc']:.4f}")

    # 最佳阈值信息
    if 'best_threshold' in metrics:
        print(f"最佳阈值: {metrics['best_threshold']:.4f}")
        print(f"最佳阈值下的MCC: {metrics['best_mcc']:.4f}")

    print('=' * 60)


def evaluate_binary_classifier(model, data_loader, device, threshold=0.5, plot=False, use_best_threshold=True):
    """
    评估二分类模型性能

    参数:
    - model: PyTorch模型
    - data_loader: 数据加载器
    - device: 设备(CPU或GPU)
    - threshold: 分类阈值（如果use_best_threshold=False则使用）
    - plot: 是否绘制可视化图表
    - use_best_threshold: 是否使用MCC值寻找最佳阈值

    返回:
    - metrics: 评估指标字典
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_scores = []

    with torch.no_grad():
        for batch in data_loader:
            # 将数据移动到指定设备
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # 提取特征和标签
            atom_x = batch['atom_x']
            atom_adj_t = batch['atom_adj_t']
            residue_x = batch['residue_x']
            residue_adj_t = batch['residue_adj_t']
            a2r_map = batch['a2r_map']
            y_true = batch['y']

            # 模型预测
            outputs = model(atom_x, atom_adj_t, residue_x, residue_adj_t, a2r_map)

            # 获取概率和预测标签
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            pos_probs = probabilities[:, 1]  # 假设正类是索引1
            pred_labels = (pos_probs >= threshold).long()

            # 收集结果
            all_preds.append(pred_labels.cpu())
            all_targets.append(y_true.cpu())
            all_scores.append(pos_probs.cpu())

    # 合并所有批次的结果
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_scores = torch.cat(all_scores)

    # 如果需要，使用MCC值寻找最佳阈值
    best_threshold = threshold
    best_mcc = None

    if use_best_threshold:
        # 寻找最佳阈值
        best_threshold, best_mcc, mcc_values, thresholds = find_best_threshold_by_mcc(
            all_targets.numpy(), all_scores.numpy()
        )

        if plot:
            # 绘制MCC与阈值的关系图
            plot_mcc_thresholds(thresholds, mcc_values, best_threshold, best_mcc)

        # 使用最佳阈值重新计算预测标签
        all_preds = (all_scores >= best_threshold).long()

    # 计算评估指标
    metrics = calculate_metrics(all_targets, all_preds, all_scores, best_threshold)

    # 添加最佳阈值信息
    if use_best_threshold:
        metrics['best_threshold'] = best_threshold
        metrics['best_mcc'] = best_mcc

    # 打印指标报告
    print_metrics_report(metrics)

    # 可视化结果
    if plot:
        plot_confusion_matrix(all_targets, all_preds, labels=['Negative', 'Positive'])
        plot_roc_curve(all_targets, all_scores)
        plot_pr_curve(all_targets, all_scores)

    return metrics


if __name__ == "__main__":
    # 测试代码
    print("二分类评估指标计算模块")
    print("从demo.py导入并使用evaluate_binary_classifier函数来评估模型")

    # 示例用法:
    # from target import evaluate_binary_classifier
    # metrics = evaluate_binary_classifier(model, test_loader, device)

    # 测试寻找最佳阈值的功能
    print("\n测试MCC最佳阈值功能...")
    # 生成模拟数据
    np.random.seed(42)
    y_true = np.random.randint(0, 2, size=1000)
    y_scores = np.random.rand(1000) * 0.4 + y_true * 0.6

    # 寻找最佳阈值
    best_threshold, best_mcc, mcc_values, thresholds = find_best_threshold_by_mcc(y_true, y_scores)

    print(f"最佳阈值: {best_threshold:.4f}")
    print(f"最佳MCC值: {best_mcc:.4f}")

    # 绘制MCC与阈值的关系
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, mcc_values, 'b-')
    plt.axvline(x=best_threshold, color='r', linestyle='--',
                label=f'Best Threshold: {best_threshold:.3f}\nMCC: {best_mcc:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('MCC')
    plt.title('MCC vs. Threshold (test data)')
    plt.legend()
    plt.grid(True)
    plt.show()