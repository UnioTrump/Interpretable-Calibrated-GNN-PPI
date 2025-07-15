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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(123)

# 设置超参数
BATCH_SIZE = 16
POS_WEIGHT = torch.tensor(1)

def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        # train_list = pickle.load(f) # 正式训练时取消注释
        raw_data = pickle.load(f)
    return raw_data

train_list = load_data('/gz-data/train355-r5.5-a2.3.pkl')

# 创建完整数据集
samples_num = len(train_list)
split_num = int(0.8 * samples_num)
data_index = train_list
np.random.shuffle(data_index)
train_data = data_index[:split_num]
val_data = data_index[split_num:]

# 计算数据集中原子和残基的最大数量
atom_nodes = []     # 13512
residue_nodes = [] # 869
for _,data_dict in enumerate(train_list):
    atom_nodes.append(len(data_dict['atom_graph_node']))
    residue_nodes.append(len(data_dict['residue_graph_node']))
max_atom_nodes = max(atom_nodes)
max_residue_nodes = max(residue_nodes)

model = HierarchicalGNN(
    atom_num_nodes=max_atom_nodes,
    residue_num_nodes=max_residue_nodes,
    atom_in_channels=37,
    residue_in_channels=1024,
    hidden_channels=128,      # 降低隐藏维度以减少计算量
    out_channels=1,           # 二分类任务，输出1个logit
    atom_num_layers=2,
    residue_num_layers=4,     # 增加残基网络深度以利用门控残差结构
    heads=4,                  # GAT的头数
    dropout=0.4,              # 调整dropout
    device=device
).to(device)

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


def train(model, train_proteins, optimizer, batch_size, grad_norm=None):
    model.train()
    np.random.shuffle(train_proteins)  # 每个epoch开始时打乱训练集

    total_loss = 0
    criterion = WeightedCrossEntropy(pos_wt=POS_WEIGHT, device=device)

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
def test(model, val_proteins):
    """评估模型在给定数据集上的性能"""
    model.eval()
    criterion = WeightedCrossEntropy(pos_wt=POS_WEIGHT, device=device)
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

# 设置训练参数
num_epochs = 100
best_loss = 999
patience = 10  # 早停耐心值
patience_counter = 0
model_dir = './checkpoints'
os.makedirs(model_dir, exist_ok=True)

# 初始化优化器和学习率调度器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.6, verbose=True, min_lr=1e-6)

# 使用tqdm创建整体训练进度条
print("开始训练...")
epoch_pbar = tqdm(range(num_epochs), desc="训练进度", ncols=180)
train_losses = []
val_losses = []
for epoch in epoch_pbar:
    # 训练
    start_time = time.time()
    train_loss = train(model, train_data, optimizer, batch_size=BATCH_SIZE, grad_norm=1.0)
    train_losses.append(train_loss)
    train_time = time.time() - start_time

    # 在验证集上评估
    val_loss, metrics, best_threshold = test(model, val_data)
    val_losses.append(val_loss)
    scheduler.step(metrics['pr_auc'])

    # 更新进度条
    epoch_pbar.set_postfix({
        'lr': f'{optimizer.param_groups[0]["lr"]:.1e}',
        'train_loss': f'{train_loss:.4f}',
        'val_loss': f'{val_loss:.4f}',
        'roc_auc': f'{metrics["roc_auc"]:.4f}',
        'pr_auc': f'{metrics["pr_auc"]:.4f}',
        'best_threshold': f'{best_threshold:.4f}',
        'Save State': 'Saved' if (val_loss < best_loss) else 'Not Saved'
    })
    epoch_pbar.update(1)
    # 保存最佳准确率模型
    if val_losses[-1] < best_loss:
        best_loss = val_loss
        save_checkpoint(
            model, optimizer, epoch, train_loss, val_loss,
            os.path.join(model_dir, f'best_model_acc.pt')
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
    
    # 早停机制
    if patience_counter >= patience:
        print(f"早停! 验证准确率在{patience}个epoch内没有改善。")
        break

print("All done!")