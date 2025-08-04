from tqdm import tqdm
import pickle
import torch
import os
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_sparse import SparseTensor
from utils import WeightedCrossEntropy, calculate_metrics
from GASPPI import add_laplacian_pe, H_GNNMambaPPI
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import config

device = config.DEVICE
torch.manual_seed(config.SEED)
print(f"Current working directory: {os.getcwd()}")
print(f"Using device: {device}")

def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def prepare_sample(sample, device):
    """Prepares a single protein sample for the H_GNNMambaPPI model."""
    row_a = torch.as_tensor(sample['a_edge_index'][0], dtype=torch.long)
    col_a = torch.as_tensor(sample['a_edge_index'][1], dtype=torch.long)
    atom_adj_t = SparseTensor(
        row=row_a, col=col_a,
        sparse_sizes=(len(sample['a_node']), len(sample['a_node']))
    ).t()
    
    atom_edge_attr = None
    if 'a_edge_feat' in sample and sample['a_edge_feat'] is not None:
        atom_edge_attr = torch.as_tensor(sample['a_edge_feat'], dtype=torch.float)
        atom_edge_attr = atom_edge_attr / config.ATOM_DISTANCE_THRESHOLD

    row_r = torch.as_tensor(sample['r_edge_index'][0], dtype=torch.long)
    col_r = torch.as_tensor(sample['r_edge_index'][1], dtype=torch.long)
    residue_adj_t = SparseTensor(
        row=row_r, col=col_r,
        sparse_sizes=(len(sample['r_node']), len(sample['r_node']))
    ).t()
    
    residue_edge_attr = None
    if 'r_edge_feat' in sample and sample['r_edge_feat'] is not None:
        residue_edge_attr = torch.as_tensor(sample['r_edge_feat'], dtype=torch.float)
        residue_edge_attr = residue_edge_attr / config.RESIDUE_DISTANCE_THRESHOLD

    label_list = [int(char) for char in sample['label']]
    label_tensor = torch.as_tensor(label_list, dtype=torch.float)

    data = Data(
        atom_x=torch.as_tensor(sample['a_node'], dtype=torch.float),
        atom_adj_t=atom_adj_t,
        atom_edge_attr=atom_edge_attr,
        atom_to_residue_map=torch.as_tensor(sample['a2r_map'], dtype=torch.long),
        residue_x=torch.as_tensor(sample['r_node'], dtype=torch.float),
        residue_adj_t=residue_adj_t,
        residue_edge_attr=residue_edge_attr,
        y=label_tensor,
    )

    pe_edge_index = torch.as_tensor(sample['r_edge_index'], dtype=torch.long)
    residue_graph_for_pe = Data(num_nodes=data.residue_x.size(0), edge_index=pe_edge_index.contiguous())
    residue_graph_for_pe = add_laplacian_pe(residue_graph_for_pe, pe_dim=config.PE_DIM)
    data.lap_pe = residue_graph_for_pe.lap_pe

    return data.to(device)

def train(model, train_proteins, optimizer, cfg):
    model.train()
    np.random.shuffle(train_proteins)

    total_loss = 0
    criterion = WeightedCrossEntropy(pos_wt=torch.tensor(cfg['POS_WEIGHT'], device=device), device=device)

    for i in range(0, len(train_proteins), cfg['BATCH_SIZE']):
        batch_proteins = train_proteins[i:i + cfg['BATCH_SIZE']]
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
                    print(f"Skipping protein due to size mismatch: {e}")
                    continue
                else:
                    raise e
        
        if cfg['GRAD_NORM'] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['GRAD_NORM'])

        optimizer.step()
        total_loss += batch_loss_sum

    return total_loss / len(train_proteins)

@torch.no_grad()
def test(model, val_proteins, cfg):
    model.eval()
    criterion = WeightedCrossEntropy(pos_wt=torch.tensor(cfg['POS_WEIGHT'], device=device), device=device)
    total_loss = 0
    all_probs, all_targets = [], []
    
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
                continue
            else:
                raise e

    if not all_targets:
        print("\n[Error] All validation samples were skipped. Cannot calculate metrics.")
        return 0.0, {'pr_auc': 0.0, 'roc_auc': 0.0, 'f1': 0.0}

    avg_loss = total_loss / len(val_proteins)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    all_probs_tensor = torch.cat(all_probs, dim=0)

    # For pre-training, we can use a default threshold of 0.5
    metrics = calculate_metrics(y_true=all_targets_tensor, y_scores=all_probs_tensor, threshold=0.5)
    
    return avg_loss, metrics

def plot_loss_curves(train_losses, val_losses, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 7))
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
    plt.title('Pre-training: Training and Validation Loss Curves')
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
    print(f"Pre-training loss curve saved to: {save_path}")
    plt.close()

def main():
    cfg = config.PRETRAIN
    print("--- Starting Pre-training ---")
    print(f"Loading pre-training data from: {cfg['DATA_PATH']}")

    all_proteins = load_data(cfg['DATA_PATH'])
    np.random.seed(config.SEED)
    np.random.shuffle(all_proteins)
    
    split_index = int(len(all_proteins) * 0.9) # Using 90/10 split for pre-training
    train_data = all_proteins[:split_index]
    val_data = all_proteins[split_index:]
    print(f'Training samples: {len(train_data)}')
    print(f'Validation samples: {len(val_data)}')

    sample_data = all_proteins[0]
    atom_in_channels = sample_data['a_node'].shape[1]
    residue_in_channels = sample_data['r_node'].shape[1]
    atom_edge_dim = sample_data['a_edge_feat'].shape[1] if 'a_edge_feat' in sample_data and sample_data['a_edge_feat'] is not None else 1
    residue_edge_dim = sample_data['r_edge_feat'].shape[1] if 'r_edge_feat' in sample_data and sample_data['r_edge_feat'] is not None else 1

    model = H_GNNMambaPPI(
        atom_in_channels=atom_in_channels,
        residue_in_channels=residue_in_channels,
        pe_dim=config.PE_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_atom_layers=config.NUM_ATOM_LAYERS,
        num_residue_layers=config.NUM_RESIDUE_LAYERS,
        out_channels=config.OUT_CHANNELS,
        mamba_d_state=config.MAMBA_D_STATE,
        mamba_d_conv=config.MAMBA_D_CONV,
        mamba_expand=config.MAMBA_EXPAND,
        dropout=config.DROPOUT,
        heads=config.HEADS,
        atom_edge_dim=atom_edge_dim,
        residue_edge_dim=residue_edge_dim
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['LEARNING_RATE'], weight_decay=cfg['WEIGHT_DECAY'])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['SCHEDULER_T_MAX'], eta_min=cfg['SCHEDULER_ETA_MIN'])

    best_val_loss = float('inf')
    patience_counter = 0
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    print("Starting pre-training loop...")
    train_losses, val_losses = [], []
    epoch_pbar = tqdm(range(cfg['EPOCHS']), desc="Pre-training Progress", ncols=180)

    for epoch in epoch_pbar:
        train_loss = train(model, train_data, optimizer, cfg)
        train_losses.append(train_loss)

        val_loss, metrics = test(model, val_data, cfg)
        val_losses.append(val_loss)
        
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), cfg['MODEL_SAVE_PATH'])
            print(f"\nEpoch {epoch+1}: Validation loss improved to {val_loss:.4f}. Model saved to {cfg['MODEL_SAVE_PATH']}")
        else:
            patience_counter += 1

        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}',
            'Val PR_AUC': f'{metrics["pr_auc"]:.4f}', 'Val ROC_AUC': f'{metrics["roc_auc"]:.4f}',
            'Patience': f'{patience_counter}/{cfg["PATIENCE"]}'
        })

        if patience_counter >= cfg['PATIENCE']:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    plot_save_path = os.path.join(config.PLOT_DIR, f'{config.PROJECT_NAME}_pretrain_loss_curve.png')
    plot_loss_curves(train_losses, val_losses, save_path=plot_save_path)

if __name__ == '__main__':
    main() 