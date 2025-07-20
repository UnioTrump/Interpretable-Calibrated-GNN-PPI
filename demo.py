from tqdm import tqdm
import pickle
import torch
import os
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_sparse import SparseTensor
from utils import WeightedCrossEntropy, calculate_metrics, find_best_threshold_by_f_beta
from GASPPI import add_gaussian_edge_weights, add_laplacian_pe, DualStreamPPI, FeatureStreamOnlyPPI
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import config

device = config.DEVICE
torch.manual_seed(config.SEED)

def load_data(pkl_path):
    return sum((pickle.load(open(os.path.join(pkl_path, f), 'rb'))
               for f in os.listdir(pkl_path) if f.endswith('.pkl')), [])

def prepare_sample(sample, device):
    atom_graph_for_weights = Data(
        x=torch.FloatTensor(sample['atom_graph_node']),
        edge_index=torch.LongTensor(sample['atom_graph_edge']),
    )
    residue_graph_for_weights = Data(
        x=torch.FloatTensor(sample['residue_graph_node']),
        edge_index=torch.LongTensor(sample['residue_graph_edge']),
    )

    atom_graph_for_weights = add_gaussian_edge_weights(atom_graph_for_weights, sigma=config.GAUSSIAN_SIGMA)
    residue_graph_for_weights = add_gaussian_edge_weights(residue_graph_for_weights, sigma=config.GAUSSIAN_SIGMA)

    atom_adj_t = SparseTensor(
        row=atom_graph_for_weights.edge_index[0], col=atom_graph_for_weights.edge_index[1],
        value=atom_graph_for_weights.edge_attr,
        sparse_sizes=(len(atom_graph_for_weights.x), len(atom_graph_for_weights.x))
    ).t()

    residue_adj_t = SparseTensor(
        row=residue_graph_for_weights.edge_index[0], col=residue_graph_for_weights.edge_index[1],
        value=residue_graph_for_weights.edge_attr,
        sparse_sizes=(len(residue_graph_for_weights.x), len(residue_graph_for_weights.x))
    ).t()

    data = Data(
        atom_x=torch.FloatTensor(sample['atom_graph_node']),
        atom_adj_t=atom_adj_t,
        residue_x=torch.FloatTensor(sample['residue_graph_node']),
        residue_adj_t=residue_adj_t,
        edge_index=torch.LongTensor(sample['residue_graph_edge']),
        y=torch.LongTensor(sample['label']),
        atom_to_residue_map=torch.tensor(sample['a2r_map'])
    )

    residue_graph_for_pe = Data(num_nodes=data.residue_x.size(0), edge_index=data.edge_index)
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
            data = prepare_sample(protein, device)
            out = model(data)
            loss = criterion.compute_loss(out, data.y)
            batch_loss_sum += loss.item()
            loss = loss / len(batch_proteins)
            loss.backward()

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

    for val_p in val_proteins:
        data = prepare_sample(val_p, device)
        out = model(data)
        loss = criterion.compute_loss(out, data.y)
        total_loss += loss.item()
        all_probs.append(torch.sigmoid(out))
        all_targets.append(data.y.float())

    avg_loss = total_loss / len(val_proteins)
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
    atom_in_channels = sample_data['atom_graph_node'].shape[1]
    residue_in_channels = sample_data['residue_graph_node'].shape[1]

    # Temporarily switch to the ablation model for diagnostics
    model_class = DualStreamPPI
    print(f"--- DIAGNOSTIC RUN: Using {model_class.__name__} ---")

    model = model_class(
        atom_in_channels=atom_in_channels,
        residue_in_channels=residue_in_channels,
        atom_hidden_dims=config.ATOM_HIDDEN_DIMS,
        residue_hidden_dims=config.RESIDUE_HIDDEN_DIMS,
        pe_dim=config.PE_DIM,
        geo_hidden_dims=config.GEO_HIDDEN_DIMS,
        fusion_hidden_dim=config.FUSION_HIDDEN_DIM,
        out_channels=config.OUT_CHANNELS,
        dropout=config.DROPOUT,
        heads=config.HEADS
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