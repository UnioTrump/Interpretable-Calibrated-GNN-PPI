import os
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_sparse import SparseTensor
import matplotlib.pyplot as plt

import config
from GASPPI.model.enhanced_dual_stream import PPIModel
from GASPPI.utils import add_gaussian_edge_weights, add_enhanced_spectral_features
from utils.metrics import calculate_metrics, find_best_threshold_by_mcc


def load_data(pkl_path):
    data_list = []
    for pkl_file in [f for f in os.listdir(pkl_path) if f.endswith('.pkl')]:
        with open(os.path.join(pkl_path, pkl_file), 'rb') as f:
            data_list.extend(pickle.load(f))
    print(f"Loaded {len(data_list)} samples.")
    return data_list


def prepare_sample(sample, device):
    """准备单个样本，为其添加谱特征"""
    atom_graph = Data(
        x=torch.FloatTensor(sample['atom_graph_node']),
        edge_index=torch.LongTensor(sample['atom_graph_edge']),
    )
    residue_graph = Data(
        x=torch.FloatTensor(sample['residue_graph_node']),
        edge_index=torch.LongTensor(sample['residue_graph_edge']),
    )

    atom_graph = add_gaussian_edge_weights(atom_graph, sigma=config.GAUSSIAN_SIGMA)
    residue_graph = add_gaussian_edge_weights(residue_graph, sigma=config.GAUSSIAN_SIGMA)

    atom_adj_t = SparseTensor.from_edge_index(
        atom_graph.edge_index, atom_graph.edge_attr.squeeze(),
        sparse_sizes=(len(atom_graph.x), len(atom_graph.x))
    ).t()
    residue_adj_t = SparseTensor.from_edge_index(
        residue_graph.edge_index, residue_graph.edge_attr.squeeze(),
        sparse_sizes=(len(residue_graph.x), len(residue_graph.x))
    ).t()

    data = Data(
        atom_x=atom_graph.x,
        atom_adj_t=atom_adj_t,
        residue_x=residue_graph.x,
        residue_adj_t=residue_adj_t,
        edge_index=residue_graph.edge_index,
        y=torch.LongTensor(sample['label']),
        atom_to_residue_map=torch.tensor(sample['a2r_map'])
    )

    graph_for_pe = Data(num_nodes=data.residue_x.size(0), edge_index=data.edge_index)
    graph_for_pe = add_enhanced_spectral_features(graph_for_pe, max_eigenvectors=config.MAX_EIGENVECTORS)
    
    data.eigenvalues = graph_for_pe.eigenvalues
    data.eigenvectors = graph_for_pe.eigenvectors
    data.spectrum_info = graph_for_pe.spectrum_info
    data.lap_pe = graph_for_pe.lap_pe

    return data.to(device)


def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        logits = model(data)
        target = data.y.float()
        
        pos_weight = torch.tensor([config.POS_WEIGHT], device=config.DEVICE)
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_NORM)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate(model, data_list):
    model.eval()
    y_true_list, y_scores_list = [], []
    
    for sample in data_list:
        data = prepare_sample(sample, config.DEVICE)
        logits = model(data)
        probs = torch.sigmoid(logits)
        y_true_list.extend(data.y.cpu().numpy())
        y_scores_list.extend(probs.cpu().numpy())
    
    y_true, y_scores = np.array(y_true_list), np.array(y_scores_list)
    best_threshold = find_best_threshold_by_mcc(y_true, y_scores)
    return calculate_metrics(y_true, y_scores, best_threshold)


def plot_training_results(train_losses, val_aucs, save_path='training_curves.png'):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(train_losses, 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(val_aucs, 'r-', label='Validation AUC')
    ax2.set_ylabel('AUC', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.title('Training Loss and Validation AUC')
    plt.savefig(os.path.join(config.PLOT_DIR, save_path))
    plt.show()


def main():
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    
    print("Spectral Attention PPI Model Training & Evaluation")
    print("=" * 60)
    
    train_data = load_data(config.DATA_PATH)
    with open(config.VAL_DATA_PATH, 'rb') as f:
        val_data = pickle.load(f)
    
    train_loader = [prepare_sample(p, config.DEVICE) for p in train_data]

    model_config = {
        'atom_in_channels': train_data[0]['atom_graph_node'].shape[1],
        'residue_in_channels': train_data[0]['residue_graph_node'].shape[1],
        'atom_hidden_dims': config.ATOM_HIDDEN_DIMS,
        'residue_hidden_dims': config.RESIDUE_HIDDEN_DIMS,
        'pe_dim': config.PE_DIM,
        'geo_hidden_dims': config.GEO_HIDDEN_DIMS,
        'fusion_hidden_dim': config.FUSION_HIDDEN_DIM,
        'out_channels': config.OUT_CHANNELS,
        'heads': config.HEADS,
        'dropout': config.DROPOUT,
        'task_type': config.TASK_TYPE
    }
    model = PPIModel(**model_config).to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.PLOT_DIR, exist_ok=True)

    train_losses, val_aucs = [], []
    best_val_auc = 0

    for epoch in range(1, config.EPOCHS + 1):
        random.shuffle(train_loader)
        train_loss = train(model, train_loader, optimizer)
        train_losses.append(train_loss)

        val_metrics = evaluate(model, val_data)
        val_aucs.append(val_metrics['auc'])
        
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val AUC: {val_metrics['auc']:.4f} | Val MCC: {val_metrics['mcc']:.4f}")

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, 'best_model.pth'))
            print(f"  -> New best model saved with AUC: {best_val_auc:.4f}")

    plot_training_results(train_losses, val_aucs)
    print("\nTraining complete. Best model saved.")
    print(f"Final Best Validation AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    main() 