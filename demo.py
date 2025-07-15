from tqdm import tqdm
import pickle
import torch
import os
import time
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_sparse import SparseTensor
from utils import WeightedCrossEntropy, calculate_metrics
from utils import find_best_threshold_by_f_beta
from GASPPI import HierarchicalGNN
# from config import DefaultConfig
import matplotlib.pyplot as plt
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(123)


def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        # train_list = pickle.load(f) # 正式训练时取消注释
        raw_data = pickle.load(f)
    return raw_data


def prepare_sample(sample, device):
    """将单个样本数据转换为torch tensor并移动到指定设备"""
    p_a_node = torch.FloatTensor(sample['atom_graph_node'])
    p_a_edge = torch.LongTensor(sample['atom_graph_edge'])
    p_r_node = torch.FloatTensor(sample['residue_graph_node'])
    p_r_edge = torch.LongTensor(sample['residue_graph_edge'])
    targets = torch.LongTensor(sample['label'])
    a2r_map = torch.tensor(sample['a2r_map'])

    # 创建原子邻接张量
    atom_edge_attr = torch.ones(p_a_edge.shape[1], dtype=torch.float)
    atom_adj_t = SparseTensor(
        row=p_a_edge[0], col=p_a_edge[1],
        value=atom_edge_attr,
        sparse_sizes=(len(p_a_node), len(p_a_node))
    ).t()

    # 创建残基邻接张量
    residue_edge_attr = torch.ones(p_r_edge.shape[1], dtype=torch.float)
    residue_adj_t = SparseTensor(
        row=p_r_edge[0], col=p_r_edge[1],
        value=residue_edge_attr,
        sparse_sizes=(len(p_r_node), len(p_r_node))
    ).t()

    return (
        p_a_node.to(device), atom_adj_t.to(device),
        p_r_node.to(device), residue_adj_t.to(device),
        targets.to(device), a2r_map.to(device)
    )


def train(model, train_proteins, optimizer, batch_size, pos_weight, grad_norm=None):
    model.train()
    np.random.shuffle(train_proteins)  # 每个epoch开始时打乱训练集

    total_loss = 0
    criterion = WeightedCrossEntropy(pos_wt=pos_weight, device=device)

    # 按批次处理数据
    for i in range(0, len(train_proteins), batch_size):
        batch_proteins = train_proteins[i:i + batch_size]
        optimizer.zero_grad()  # 为新批次清零梯度

        batch_loss_sum = 0 # 用于正确记录该批次的总损失
        # 在批次内累积梯度
        for protein in batch_proteins:
            p_a_node, atom_adj_t, p_r_node, residue_adj_t, targets, a2r_map = prepare_sample(protein, device)

            # 前向传播
            out = model(p_a_node, atom_adj_t, p_r_node, residue_adj_t, a2r_map)
            loss = criterion.compute_loss(out, targets)
            batch_loss_sum += loss.item() # 为日志记录累积未经缩放的损失

            # 为了平均梯度，将损失除以批次大小
            loss = loss / len(batch_proteins)
            loss.backward()  # 累积梯度

        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)

        optimizer.step()  # 在批次结束后更新权重

        # 记录该批次的损失
        total_loss += batch_loss_sum

    return total_loss / len(train_proteins)


@torch.no_grad()
def test(model, val_proteins, pos_weight):
    """评估模型在给定数据集上的性能"""
    model.eval()
    criterion = WeightedCrossEntropy(pos_wt=pos_weight, device=device)
    total_loss = 0
    all_probs, all_targets = [], []

    for val_p in val_proteins:
        p_a_node, atom_adj_t, p_r_node, residue_adj_t, targets, a2r_map = prepare_sample(val_p, device)

        # 前向传播
        out = model(p_a_node, atom_adj_t, p_r_node, residue_adj_t, a2r_map)

        # 计算损失
        loss = criterion.compute_loss(out, targets)
        total_loss += loss.item()

        # 记录batch标签和预测值
        all_probs.append(torch.sigmoid(out))
        all_targets.append(targets.float())

    avg_loss = total_loss / len(val_proteins)

    all_targets_tensor = torch.cat(all_targets, dim=0)
    all_probs_tensor = torch.cat(all_probs, dim=0)

    threshold, f_beta = find_best_threshold_by_f_beta(
        all_targets_tensor,
        all_probs_tensor,
        num_threshold=100)
    metrics = calculate_metrics(
        y_true=all_targets_tensor,
        y_scores=all_probs_tensor,
        threshold=threshold)

    return avg_loss, metrics, threshold


def save_checkpoint(model, optimizer, epoch, loss, val_loss, filename):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'val_loss': val_loss,
    }
    torch.save(checkpoint, filename)

def plot_loss_curves(train_losses, val_losses, save_path='loss_curves.png'):
    """
    绘制并保存训练和验证损失曲线图。

    参数:
    - train_losses (list): 包含每个epoch训练损失的列表。
    - val_losses (list): 包含每个epoch验证损失的列表。
    - save_path (str): 图片保存路径。
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 找到验证损失最低点并标记
    best_val_epoch = np.argmin(val_losses)
    best_val_loss = val_losses[best_val_epoch]
    plt.axvline(x=best_val_epoch + 1, color='gray', linestyle='--', label=f'Best Val Loss Epoch: {best_val_epoch+1}')
    plt.scatter(best_val_epoch + 1, best_val_loss, marker='*', s=150, color='gold', zorder=5, label=f'Best Val Loss: {best_val_loss:.4f}')
    
    # 重新整理图例，避免重叠
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.savefig(save_path, dpi=300)
    print(f"损失曲线图已保存至: {save_path}")
    plt.close()


def main(args):
    # 使用args设置超参数
    BATCH_SIZE = args.batch_size
    POS_WEIGHT = torch.tensor(args.pos_weight).to(device)
    
    train_list = load_data(args.data_path)

    # 创建完整数据集
    samples_num = len(train_list)
    split_num = int(0.8 * samples_num)
    data_index = list(train_list)
    np.random.shuffle(data_index)
    train_data = data_index[:split_num]
    val_data = data_index[split_num:]

    # 计算数据集中原子和残基的最大数量
    if not train_list:
        print("错误: 数据列表为空。")
        return
        
    atom_nodes = [len(d['atom_graph_node']) for d in train_list]
    residue_nodes = [len(d['residue_graph_node']) for d in train_list]
    max_atom_nodes = max(atom_nodes) if atom_nodes else 0
    max_residue_nodes = max(residue_nodes) if residue_nodes else 0

    model = HierarchicalGNN(
        atom_num_nodes=max_atom_nodes,
        residue_num_nodes=max_residue_nodes,
        atom_in_channels=37,
        residue_in_channels=1024,
        hidden_channels=256,
        out_channels=1,
        atom_num_layers=4,
        residue_num_layers=4,
        dropout=0.4,
        pool_size=2,
        buffer_size=500,
        device=device
    ).to(device)

    # 设置训练参数
    num_epochs = args.epochs
    best_pr_auc = 0.0
    patience = args.patience
    patience_counter = 0
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    # 初始化优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=True)

    # 使用tqdm创建整体训练进度条
    print("开始训练...")
    epoch_pbar = tqdm(range(num_epochs), desc="训练进度", ncols=180)
    train_losses = []
    val_losses = []

    for epoch in epoch_pbar:
        # 训练
        start_time = time.time()
        train_loss = train(model, train_data, optimizer, batch_size=BATCH_SIZE, pos_weight=POS_WEIGHT, grad_norm=1.0)
        train_losses.append(train_loss)
        train_time = time.time() - start_time

        # 在验证集上评估
        val_loss, metrics, best_threshold = test(model, val_data, pos_weight=POS_WEIGHT)
        val_losses.append(val_loss)
        # 让学习率调度器监控pr_auc
        scheduler.step(metrics['pr_auc'])

        # 更新进度条
        current_pr_auc = metrics["pr_auc"]
        epoch_pbar.set_postfix({
            'lr': f'{optimizer.param_groups[0]["lr"]:.1e}',
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'roc_auc': f'{metrics["roc_auc"]:.4f}',
            'best_pr_auc': f'{best_pr_auc:.4f}',
            'Save State': 'Saved' if (current_pr_auc > best_pr_auc) else 'Not Saved'
        })
        epoch_pbar.update(1)
        
        # 根据PR AUC保存最佳模型
        if current_pr_auc > best_pr_auc:
            best_pr_auc = current_pr_auc
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                os.path.join(model_dir, f'best_model_pr_auc.pt')
            )
            patience_counter = 0
        else:
            patience_counter += 1

        # 每10个epoch保存一次检查点
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                os.path.join(model_dir, f'model_epoch_{epoch+1}.pt')
            )
        
        # 基于PR AUC的早停机制
        if patience_counter >= patience:
            print(f"早停! 验证集PR_AUC在 {patience} 个epoch内没有改善。")
            break

    print("训练结束！")
    
    # 绘制并保存损失曲线
    figure_save_path = os.path.join(args.loss_figure_path, 'loss_curves.png')
    plot_loss_curves(train_losses, val_losses, save_path=figure_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GASPPI Training Script')
    parser.add_argument('--data_path', type=str, default='/gz-data/train355-r5.5-a2.3.pkl', help='Path to training data pkl file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for AdamW optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for AdamW optimizer')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--pos_weight', type=float, default=2.0, help='Positive class weight for loss function')
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--loss_figure_path', type=str, default='save_fig', help='Directory to save loss curve figure')
    
    args = parser.parse_args()
    
    # 创建图片保存目录
    os.makedirs(args.loss_figure_path, exist_ok=True)
    
    main(args)