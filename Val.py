import torch
import os
from utils import calculate_metrics, find_best_threshold_by_f_beta
from GASPPI import DualStreamPPI
import config
from tqdm import tqdm
from data_utils import DataLoader

device = config.DEVICE

@torch.no_grad()
def A(models, val_proteins, data_loader):
    all_prob, all_target = [], []

    for val_p in tqdm(val_proteins, desc='Testing'):
        data = data_loader.prepare_sample(val_p)

        model_features = []
        for model in models:
            with torch.no_grad():
                out = model.feat(data)
                model_features.append(out)

        features_tensor = torch.stack(model_features, dim=0)
        combined_features = features_tensor.mean(dim=0)

        with torch.no_grad():
            pred = models[0].MLP(combined_features)
            probs = torch.sigmoid(pred).squeeze()

        all_prob.append(probs.detach().cpu())
        all_target.append(data.y.float().squeeze().detach().cpu())

    all_targets_tensor = torch.cat(all_target, dim=0)
    all_probs_tensor = torch.cat(all_prob, dim=0)

    threshold, _ = find_best_threshold_by_f_beta(all_targets_tensor, all_probs_tensor, num_threshold=100)
    metrics = calculate_metrics(y_true=all_targets_tensor, y_scores=all_probs_tensor, threshold=threshold)

    return metrics
def main():
    for index, seed in enumerate(config.SEED):
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        data_loader = DataLoader(device=device)
        all_proteins = data_loader.load_data(config.VAL_DATA_PATH)

        data_info = data_loader.get_data_info(all_proteins[0])
        atom_in_channels = data_info['atom_in_channels']
        residue_in_channels = data_info['residue_in_channels']

        model_class = DualStreamPPI
        models=[]
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

        best_model_path = os.path.join(config.TUNING_MODEL, f'{index}_best_model.pth')
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        models.append(model)

    metrics = A(models, all_proteins, data_loader)
    print('-' * 20)
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            print(f"  {key.replace('_', ' ').title()}: {val:.4f}")
        elif isinstance(val, dict):
            print(f"  {key.replace('_', ' ').title()}:")
            for sub_key, sub_val in val.items():
                print(f"    {sub_key.upper()}: {sub_val}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {val}")
    print('-' * 20)

if __name__ == '__main__':
    main()