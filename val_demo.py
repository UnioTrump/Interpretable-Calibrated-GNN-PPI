import os
import torch
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

from utils import ProteinData, PPIDataLoader
from GASPPI import HierarchicalGNN


def load_data(path):
    """加载数据集"""
    with open(path, 'rb') as f:
        return pickle.load(f)


def create_model(dataset, device):
    """创建模型"""
    # 获取最大节点数
    max_atom_nodes = max(samples['atom_graph'].x.size(0) for samples in dataset.samples)
    max_residue_nodes = max(samples['residue_graph'].x.size(0) for samples in dataset.samples)

    # 获取特征维度
    atom_in_channels = dataset.samples[0]['atom_graph'].x.size(1)
    residue_in_channels = dataset.samples[0]['residue_graph'].x.size(1)

    print(f"创建模型: 原子节点={max_atom_nodes}, 残基节点={max_residue_nodes}")
    print(f"原子特征维度={atom_in_channels}, 残基特征维度={residue_in_channels}")

    # 创建模型
    model = HierarchicalGNN(
        atom_num_nodes=max_atom_nodes,
        residue_num_nodes=max_residue_nodes,
        atom_in_channels=37,
        residue_in_channels=1024,
        hidden_channels=256,
        hidden_heads=8,
        out_channels=128,
        out_heads=1,
        atom_num_layers=3,
        residue_num_layers=3,
        num_blocks=2,
        pool_size=2,
        buffer_size=500,
        dropout=0.3,
        device=device
    ).to(device)

    return model


def evaluate_model(model, loader, device, threshold=0.5):
    """评估模型性能"""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    all_probs, all_labels, losses = [], [], []

    with torch.no_grad():
        for batch in loader:
            # 将所有张量移动到指定设备
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                    
            # 正确地提取批次数据
            outputs = model(
                batch['atom_x'], 
                batch['atom_adj_t'], 
                batch['residue_x'], 
                batch['residue_adj_t'], 
                batch['a2r_map']
            )
            
            # 确保输出形状兼容交叉熵损失，与demo.py保持一致
            if outputs.dim() == 1 or (outputs.dim() == 2 and outputs.shape[1] == 1):
                # 将单输出转换为二分类的情况
                outputs = torch.cat([1-outputs, outputs], dim=1)
                
            loss = criterion(outputs, batch['y'])
            
            # 确保输出有适当的维度进行softmax
            if outputs.shape[1] >= 2:  # 如果输出至少有2个类
                probs = torch.softmax(outputs, dim=1)[:, 1]
            else:
                # 处理单输出的情况 - 先转换为概率再使用
                probs = torch.sigmoid(outputs.squeeze())
                
            losses.append(loss.item())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch['y'].cpu().numpy())

    return np.array(all_probs), np.array(all_labels), np.mean(losses)


def find_best_threshold_by_mcc(y_true, y_scores, num_thresholds=100):
    """通过MCC值寻找最佳阈值"""
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


def plot_confusion_matrix(y_true, y_pred, save_path):
    """绘制并保存混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')

    # 添加标签
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    # 添加数值
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j],
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true, y_scores, save_path):
    """绘制并保存ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_dataset(data_path, device, weights_path=None, batch_size=8, result_dir='evaluation_results'):
    """直接在数据集上评估模型"""
    # 创建结果目录
    os.makedirs(result_dir, exist_ok=True)

    # 加载数据
    print(f"加载数据: {data_path}")
    raw_data = load_data(data_path)
    dataset = ProteinData(raw_data)
    print(f"数据集大小: {len(dataset)}")

    # 创建数据加载器
    loader = PPIDataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    model = create_model(dataset, device)
    
    # 加载预训练权重
    if weights_path and os.path.exists(weights_path):
        print(f"加载模型权重: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device)
        # 如果模型是以checkpoint格式保存的(包含model_state_dict)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("模型权重加载成功!")
    else:
        print("check path", os.path.exists(weights_path))
        print("警告: 使用随机初始化权重进行评估")

    # 评估模型
    print("开始评估模型...")
    probs, labels, loss = evaluate_model(model, loader, device)

    # 寻找基于MCC的最佳阈值
    best_threshold, best_mcc = find_best_threshold_by_mcc(labels, probs)
    print(f"MCC最佳阈值: {best_threshold:.4f}, 最佳MCC值: {best_mcc:.4f}")

    # 使用基于MCC的阈值计算所有指标
    metrics = calculate_metrics(labels, probs, best_threshold)
    metrics.update({
        'loss': loss,
        'best_threshold': best_threshold,
        'best_mcc': best_mcc,
        'samples_count': len(labels)
    })

    # 保存结果
    with open(os.path.join(result_dir, 'metrics.txt'), 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    # 生成并保存图表
    preds = (probs >= best_threshold).astype(int)
    plot_confusion_matrix(labels, preds, os.path.join(result_dir, 'confusion_matrix.png'))
    plot_roc_curve(labels, probs, os.path.join(result_dir, 'roc_curve.png'))

    # 打印结果摘要
    print("\n===== 评估结果 =====")
    print(f"基于MCC的最佳阈值: {best_threshold:.4f}")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1分数: {metrics['f1_score']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"混淆矩阵:")
    print(f"  真正例 (TP): {metrics['tp']}")
    print(f"  假正例 (FP): {metrics['fp']}")
    print(f"  真负例 (TN): {metrics['tn']}")
    print(f"  假负例 (FN): {metrics['fn']}")
    
    print(f"\n评估结果已保存至: {result_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PPI模型评估')
    parser.add_argument('--data_path', default='/gz-data/Test60.pkl', help='数据文件路径')
    parser.add_argument('--weights_path', default='./checkpoints/best_model_loss.pt', help='预训练模型权重路径(.pt文件)')
    parser.add_argument('--batch_size', type=int, default=8, help='批处理大小')
    parser.add_argument('--result_dir', default='evaluation_results', help='结果保存目录')
    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 执行评估
    evaluate_dataset(
        args.data_path,
        device,
        weights_path=args.weights_path,
        batch_size=args.batch_size,
        result_dir=args.result_dir
    )


if __name__ == '__main__':
    main()