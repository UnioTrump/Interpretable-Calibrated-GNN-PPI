from tqdm import tqdm
import pickle
import torch
import os
import time
import numpy as np
from utils import WeightedCrossEntropy, calculate_metrics
from utils import find_best_threshold_by_f_beta, find_best_threshold_by_mcc
from GASPPI import HierarchicalGNN
# from config import DefaultConfig
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(123)

'''
def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        raw_data = pickle.load(f)
    print(f"Loaded data from {pkl_path}")
    return raw_data
'''
def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        train_list = pickle.load(f)
    return train_list

train_list = load_data('/gz-data/train355-r5.5-a2.3.pkl')

# 创建完整数据集
'''
full_dataset = ProteinData(train_list)
'''
samples_num = len(train_list)
split_num = int(0.8 * samples_num)
data_index = train_list
np.random.shuffle(data_index)
train_data = data_index[:split_num]
val_data = data_index[split_num:]
'''
# 划分数据集
def split_dataset(dataset, train_ratio=0.8, val_ratio=0.2, random_seed=123):
    """将数据集分割为训练集、验证集"""
    assert train_ratio + val_ratio == 1.0, "比例之和必须为1"
    
    # 设置随机种子以确保可重现性
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # 打乱数据集索引
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    # 计算分割点
    train_end = int(dataset_size * train_ratio)
    val_end = train_end + int(dataset_size * val_ratio)
    
    # 获取各部分索引
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    
    # 创建子数据集
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]

# 创建新的数据集实例
    train_set = ProteinData.__new__(ProteinData)
    train_set.samples = train_dataset
    
    val_set = ProteinData.__new__(ProteinData)
    val_set.samples = val_dataset
    print(f"数据集分割完成: 训练集 {len(train_set)}，验证集 {len(val_set)}")
    return train_set, val_set

# 划分数据集
train_dataset, val_dataset = split_dataset(full_dataset)

# 创建数据加载器
train_loader = PPIDataLoader(train_data, batch_size=16, shuffle=True)
val_loader = PPIDataLoader(val_data, batch_size=16, shuffle=True)
'''
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
    hidden_channels=256,
    hidden_heads=8,
    out_channels=128,
    out_heads=1,
    atom_num_layers=2,
    residue_num_layers=2,
    num_blocks=2,
    pool_size=2,
    buffer_size=500,
    dropout=0.3,
    device=device
).to(device)

POS_WEIGHT = 1 / torch.tensor(np.sqrt(5.39734))  # 定义全局正样本权重，提升稳定性.这是全部样本的正负样本比值
def train(run, model, train_proteins, optimizer, grad_norm=None, delta=0.5):
    model.train()

    e_loss = 0
    '''还未使用POS_WEIGHT'''
    criterion = WeightedCrossEntropy(pos_wt=1/POS_WEIGHT, device=device)
    # 移除内部进度条
    for train_p in train_proteins:
        train_p_a_node = torch.FloatTensor(train_p['atom_graph_node'])
        train_p_a_edge = torch.LongTensor(train_p['atom_graph_edge'])
        train_p_r_node = torch.FloatTensor(train_p['residue_graph_node'])
        train_p_r_edge = torch.LongTensor(train_p['residue_graph_edge'])
        targets = torch.LongTensor(train_p['label'])
        a2r_map = torch.tensor(train_p['a2r_map'])

        train_p_r_node = train_p_r_node.to(device)
        train_p_r_edge = train_p_r_edge.to(device)
        targets = targets.to(device)
        train_p_a_node = train_p_a_node.to(device)
        train_p_a_edge = train_p_a_edge.to(device)
        a2r_map = a2r_map.to(device)
        '''
        mask = batch['train_mask']
        if mask.sum() == 0:
            continue

        optimizer.zero_grad()
        
        atom_x = batch['atom_x']
        atom_adj_t = batch['atom_adj_t']
        residue_x = batch['residue_x']
        residue_adj_t = batch['residue_adj_t']
        a2r_map = batch['a2r_map']
        targets = batch['y']
        '''
        # 前向传播
        optimizer.zero_grad()
        out = model(train_p_a_node, train_p_a_edge, train_p_r_node, train_p_r_edge, a2r_map)
        loss = criterion.compute_loss(out, targets)
        loss.backward()
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()

        b_loss = loss.item()
        e_loss += b_loss
        e_loss /= len(train_proteins)
        '''
        batch_loss = float(loss) * mask.sum().item()
        total_loss += batch_loss
        total_examples += mask.sum().item()
        '''
    return e_loss

@torch.no_grad()
def test(model, val_proteins):
    """评估模型在给定数据集上的性能"""
    model.eval()
    criterion = WeightedCrossEntropy(pos_wt=1/POS_WEIGHT, device=device)
    b_loss = e_loss = 0
    all_probs, all_targets = [], []

    for val_p in val_proteins:
        val_p_a_node = torch.FloatTensor(val_p['atom_graph_node'])
        val_p_a_edge = torch.LongTensor(val_p['atom_graph_edge'])
        val_p_r_node = torch.FloatTensor(val_p['residue_graph_node'])
        val_p_r_edge = torch.LongTensor(val_p['residue_graph_edge'])
        targets = torch.LongTensor(val_p['label'])
        a2r_map = torch.tensor(val_p['a2r_map'])

        val_p_r_node = val_p_r_node.to(device)
        val_p_r_edge = val_p_r_edge.to(device)
        targets = targets.to(device)
        val_p_a_node = val_p_a_node.to(device)
        val_p_a_edge = val_p_a_edge.to(device)
        a2r_map = a2r_map.to(device)

        # 前向传播
        optimizer.zero_grad()
        out = model(val_p_a_node, val_p_a_edge, val_p_r_node, val_p_r_edge, a2r_map)

        # 计算损失
        loss = criterion.compute_loss(out, targets)

        b_loss = loss.item()
        e_loss += b_loss
        e_loss /= len(val_proteins)

        # 记录batch标签和预测值
        all_probs.append(torch.sigmoid(out))
        all_targets.append(targets.float())

    threshold, f_beta = find_best_threshold_by_mcc(
        torch.cat(all_targets,dim=0),
        torch.cat(all_probs,dim=0))
    metrics = calculate_metrics(
        torch.cat(all_targets, dim=0),
        torch.cat(all_probs, dim=0),
        threshold)

    return e_loss, metrics, threshold

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

# 初始化优化器
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3, nesterov=True)
# 使用tqdm创建整体训练进度条
print("开始训练...")
epoch_pbar = tqdm(range(num_epochs), desc="训练进度", ncols=160)
train_losses = []
val_losses = []
for epoch in epoch_pbar:
    # 训练
    start_time = time.time()
    train_loss = train(0, model, train_data, optimizer, grad_norm=1.0)
    train_losses.append(train_loss)
    train_time = time.time() - start_time

    # 在验证集上评估
    val_loss, metrics, best_threshold = test(model, val_data)
    val_losses.append(val_loss)

    # 更新进度条
    epoch_pbar.set_postfix({
        'train_loss': f'{train_loss:.4f}',
        'val_loss': f'{val_loss:.4f}',
        'recall': f'{metrics["recall"]:.4f}',
        'precision': f'{metrics["precision"]:.4f}',
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