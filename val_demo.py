import torch
import os
from utils import calculate_metrics, find_best_threshold_by_f_beta
from GASPPI import DualStreamPPI
import config
from tqdm import tqdm
from data_utils import DataLoader

device = config.DEVICE
torch.manual_seed(config.SEED)

@torch.no_grad()
def test(model, val_proteins, data_loader):
    model.eval()
    all_probs, all_targets = [], []

    for val_p in tqdm(val_proteins, desc='Testing'):
        data = data_loader.prepare_sample(val_p)
        out = model(data)
        all_probs.append(torch.sigmoid(out))
        all_targets.append(data.y.float())

    all_targets_tensor = torch.cat(all_targets, dim=0)
    all_probs_tensor = torch.cat(all_probs, dim=0)

    threshold, _ = find_best_threshold_by_f_beta(all_targets_tensor, all_probs_tensor, num_threshold=100)
    metrics = calculate_metrics(y_true=all_targets_tensor, y_scores=all_probs_tensor, threshold=threshold)

    return metrics

def main():

    data_loader = DataLoader(device=device)
    all_proteins = data_loader.load_data(config.VAL_DATA_PATH)
    print('FUCKING Load Done!!!!!')

    data_info = data_loader.get_data_info(all_proteins[0])
    atom_in_channels = data_info['atom_in_channels']
    residue_in_channels = data_info['residue_in_channels']

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

    best_model_path = os.path.join(config.TUNING_MODEL, 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print('Load the FUCKING model Done!!!')

    metrics = test(model, all_proteins, data_loader)
    print('-'*20)
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            print(f"  {key.replace('_', ' ').title()}: {val:.4f}")
        elif isinstance(val, dict):
            print(f"  {key.replace('_', ' ').title()}:")
            for sub_key, sub_val in val.items():
                print(f"    {sub_key.upper()}: {sub_val}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {val}")
    print('-'*20)

if __name__ == '__main__':
    main()