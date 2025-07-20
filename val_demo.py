import os
import torch
import pickle
from tqdm import tqdm

from utils import calculate_metrics, find_best_threshold_by_f_beta
from GASPPI import add_gaussian_edge_weights, add_laplacian_pe, DualStreamPPI, FeatureStreamOnlyPPI
from torch_geometric.data import Data
from torch_sparse import SparseTensor
import config

device = config.DEVICE

def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        data_list = pickle.load(f)
    print(f"Loaded {len(data_list)} samples for evaluation.")
    return data_list

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


@torch.no_grad()
def evaluate(model, data_list, device):
    model.eval()
    all_probs = []
    all_targets = []

    print("Starting model evaluation...")
    for sample in tqdm(data_list, desc="Evaluating"):
        data = prepare_sample(sample, device)
        out = model(data)
        all_probs.append(torch.sigmoid(out))
        all_targets.append(data.y.float())

    all_targets_tensor = torch.cat(all_targets, dim=0).cpu()
    all_probs_tensor = torch.cat(all_probs, dim=0).cpu()

    print("Evaluation complete, calculating metrics...")
    threshold, _ = find_best_threshold_by_f_beta(all_targets_tensor, all_probs_tensor, num_threshold=100)
    metrics = calculate_metrics(y_true=all_targets_tensor, y_scores=all_probs_tensor, threshold=threshold)
    
    return metrics

def main():
    # Load the validation dataset
    val_data = load_data(config.VAL_DATA_PATH)
    if not val_data:
        print("Validation data list is empty. Exiting.")
        return

    # Use the first sample to determine model dimensions
    sample_data = val_data[0]
    atom_in_channels = sample_data['atom_graph_node'].shape[1]
    residue_in_channels = sample_data['residue_graph_node'].shape[1]

    # Temporarily switch to the ablation model for diagnostics
    model_class = FeatureStreamOnlyPPI
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
        heads=config.HEADS,
        dropout=config.DROPOUT
    ).to(device)

    model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'. Please run demo.py first.")
        return
        
    print(f"Loading model weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model weights loaded successfully!")

    final_metrics = evaluate(model, val_data, device)
    
    print("\n--- Model Performance Report (Validation Set) ---")
    print("-------------------------------------------------")
    for key, val in final_metrics.items():
        if isinstance(val, (int, float)):
             print(f"  {key.replace('_', ' ').title()}: {val:.4f}")
        elif isinstance(val, dict):
            print(f"  {key.replace('_', ' ').title()}:")
            for sub_key, sub_val in val.items():
                print(f"    {sub_key.upper()}: {sub_val}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {val}")
    print("-------------------------------------------------\n")

if __name__ == '__main__':
    main()