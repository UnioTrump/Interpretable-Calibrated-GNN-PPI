from tqdm import tqdm
import pickle
import torch
import os
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_sparse import SparseTensor
from utils import WeightedCrossEntropy, calculate_metrics, find_best_threshold_by_f_beta
from GASPPI import add_laplacian_pe, H_GNNMambaPPI
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import config

device = config.DEVICE
torch.manual_seed(config.SEED)
# 输出当前工作目录
print(os.getcwd())

def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def prepare_sample(sample, device):
    """Prepares a single protein sample for the H_GNNMambaPPI model."""
    
    # Atom-level graph
    # 使用 torch.as_tensor 替换 torch.tensor 来彻底解决 UserWarning
    row_a = torch.as_tensor(sample['a_edge_index'][0], dtype=torch.long)
    col_a = torch.as_tensor(sample['a_edge_index'][1], dtype=torch.long)
    atom_adj_t = SparseTensor(
        row=row_a,
        col=col_a,
        sparse_sizes=(len(sample['a_node']), len(sample['a_node']))
    ).t()
    
    atom_edge_attr = None
    if 'a_edge_feat' in sample:
        atom_edge_attr = torch.as_tensor(sample['a_edge_feat'], dtype=torch.float)

    # Residue-level graph
    row_r = torch.as_tensor(sample['r_edge_index'][0], dtype=torch.long)
    col_r = torch.as_tensor(sample['r_edge_index'][1], dtype=torch.long)
    residue_adj_t = SparseTensor(
        row=row_r,
        col=col_r,
        sparse_sizes=(len(sample['r_node']), len(sample['r_node']))
    ).t()
    
    residue_edge_attr = None
    if 'r_edge_feat' in sample:
        residue_edge_attr = torch.as_tensor(sample['r_edge_feat'], dtype=torch.float)

    label_list = [int(char) for char in sample['label']]
    label_tensor = torch.as_tensor(label_list, dtype=torch.float)

    # --- New: Generate residue_seq_ids from one-hot encoded residue features ---
    # Assuming the first 21 features of r_node are one-hot encoding for amino acids
    residue_one_hot = torch.as_tensor(sample['r_node'][:, :21], dtype=torch.float)
    residue_seq_ids = torch.argmax(residue_one_hot, dim=1).long()

    data = Data(
        atom_x=torch.as_tensor(sample['a_node'], dtype=torch.float),
        atom_adj_t=atom_adj_t,
        atom_edge_attr=atom_edge_attr,
        atom_to_residue_map=torch.as_tensor(sample['a2r_map'], dtype=torch.long),
        residue_x=torch.as_tensor(sample['r_node'], dtype=torch.float),
        residue_adj_t=residue_adj_t,
        residue_edge_attr=residue_edge_attr,
        residue_seq_ids=residue_seq_ids,  # Add the new attribute here
        y=label_tensor,
    )

    # Add Laplacian Positional Encodings for residue graph
    pe_edge_index = torch.as_tensor(sample['r_edge_index'], dtype=torch.long)
    residue_graph_for_pe = Data(num_nodes=data.residue_x.size(0), edge_index=pe_edge_index.contiguous())
    residue_graph_for_pe = add_laplacian_pe(residue_graph_for_pe, pe_dim=config.PE_DIM)
    data.lap_pe = residue_graph_for_pe.lap_pe

    return data.to(device)

def train(model, train_proteins, optimizer):
    model.train()
    np.random.shuffle(train_proteins)

    total_loss = 0
    criterion = WeightedCrossEntropy(pos_wt=torch.tensor(config.POS_WEIGHT, device=device), device=device)

    for i in range(0, len(train_proteins), config.BATCH_SIZE):
        batch_proteins = train_proteins[i:i + config.BATCH_SIZE]
        optimizer.zero_grad()

        batch_loss_sum = 0
        for protein in batch_proteins:
            try:
                data = prepare_sample(protein, device)
                out = model(data)
                loss = criterion.compute_loss(out, data.y)
                batch_loss_sum += loss.item()
                loss = loss / len(batch_proteins)
                loss.backward()
            except RuntimeError as e:
                if "Sizes of tensors must match" in str(e):
                    continue
                else:
                    raise e

        if config.GRAD_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_NORM)

        optimizer.step()
        total_loss += batch_loss_sum

    return total_loss / len(train_proteins)

@torch.no_grad()
def test(model, val_proteins):
    model.eval()
    criterion = WeightedCrossEntropy(pos_wt=torch.tensor(config.POS_WEIGHT, device=device), device=device)
    total_loss = 0
    all_probs, all_targets = [], []
    skipped_count = 0

    for val_p in val_proteins:
        try:
            data = prepare_sample(val_p, device)
            out = model(data)
            loss = criterion.compute_loss(out, data.y)
            total_loss += loss.item()
            all_probs.append(torch.sigmoid(out))
            all_targets.append(data.y.float())
        except RuntimeError as e:
            if "Sizes of tensors must match" in str(e):
                skipped_count += 1
                continue
            else:
                raise e

    if len(val_proteins) == skipped_count:
        print("\n[错误] 所有验证样本都因不一致而被跳过，无法计算指标。")
        dummy_metrics = {
            'pr_auc': 0.0, 'roc_auc': 0.0, 'f1': 0.0, 
            'recall': 0.0, 'precision': 0.0, 'accuracy': 0.0
        }
        return 0.0, dummy_metrics, 0.5

    avg_loss = total_loss / (len(val_proteins) - skipped_count)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    all_probs_tensor = torch.cat(all_probs, dim=0)

    threshold, _ = find_best_threshold_by_f_beta(all_targets_tensor, all_probs_tensor, num_threshold=100)
    metrics = calculate_metrics(y_true=all_targets_tensor, y_scores=all_probs_tensor, threshold=threshold)

    return avg_loss, metrics, threshold

def plot_loss_curves(train_losses, val_losses, save_path='loss_curves.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if val_losses:
        best_val_epoch = np.argmin(val_losses)
        best_val_loss = val_losses[best_val_epoch]
        plt.axvline(x=best_val_epoch + 1, color='gray', linestyle='--')
        plt.scatter(best_val_epoch + 1, best_val_loss, marker='*', s=150, color='gold', zorder=5, label=f'Best Val Loss: {best_val_loss:.4f} at Epoch {best_val_epoch+1}')
    
    plt.legend()
    plt.savefig(save_path, dpi=300)
    print(f"Loss curve saved to: {save_path}")
    plt.close()

def main():

    all_proteins = load_data(config.DATA_PATH)

    np.random.seed(config.SEED)
    np.random.shuffle(all_proteins)
    
    split_index = int(len(all_proteins) * 0.8)
    train_data = all_proteins[:split_index]
    val_data = all_proteins[split_index:]
    print(f'Training samples: {len(train_data)}')
    print(f'Validation samples: {len(val_data)}')

    sample_data = all_proteins[0]
    atom_in_channels = sample_data['a_node'].shape[1]
    residue_in_channels = sample_data['r_node'].shape[1]
    atom_edge_dim = sample_data['a_edge_feat'].shape[1]
    residue_edge_dim = sample_data['r_edge_feat'].shape[1]

    model_class = H_GNNMambaPPI
    print(f"--- Using Hierarchical SOTA Model: {model_class.__name__} ---")

    model = model_class(
        atom_in_channels=atom_in_channels,
        residue_in_channels=residue_in_channels,
        pe_dim=config.PE_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_atom_layers=config.NUM_ATOM_LAYERS,
        num_residue_layers=config.NUM_RESIDUE_LAYERS,
        num_seq_layers=config.NUM_SEQ_LAYERS,
        vocab_size=config.VOCAB_SIZE,
        out_channels=config.OUT_CHANNELS,
        mamba_d_state=config.MAMBA_D_STATE,
        mamba_d_conv=config.MAMBA_D_CONV,
        mamba_expand=config.MAMBA_EXPAND,
        dropout=config.DROPOUT,
        heads=config.HEADS,
        atom_edge_dim=atom_edge_dim,
        residue_edge_dim=residue_edge_dim
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.SCHEDULER_T_MAX, eta_min=config.SCHEDULER_ETA_MIN)

    best_pr_auc = 0.0
    patience_counter = 0
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    best_model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    
    print("Starting training...")
    train_losses, val_losses = [], []
    epoch_pbar = tqdm(range(config.EPOCHS), desc="Training Progress", ncols=180)

    for epoch in epoch_pbar:
        train_loss = train(model, train_data, optimizer)
        train_losses.append(train_loss)

        val_loss, metrics, best_threshold = test(model, val_data)
        val_losses.append(val_loss)
        
        pr_auc = metrics['pr_auc']
        scheduler.step()

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}',
            'Val PR_AUC': f'{pr_auc:.4f}', 'Val ROC_AUC': f'{metrics["roc_auc"]:.4f}',
            'Patience': f'{patience_counter}/{config.PATIENCE}'
        })

        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    plot_save_path = os.path.join(config.PLOT_DIR, f'{config.PROJECT_NAME}_loss_curve.png')
    plot_loss_curves(train_losses, val_losses, save_path=plot_save_path)

if __name__ == '__main__':
    main()