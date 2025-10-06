import torch
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import calculate_metrics, find_best_threshold_by_f_beta, save_metrics_to_txt
from GASPPI import DualStreamPPI
import config
from tqdm import tqdm
from data_utils import DataLoader
import numpy as np
import argparse

device = config.DEVICE

@torch.no_grad()
def A(model, val_proteins, data_loader,Dset_name, visualize=True, ):
    all_prob, all_target = [], []
    extracted_features, labels = [], []

    for p in tqdm(val_proteins, desc='Testing'):
        data = data_loader.prepare_sample(p)

        with torch.no_grad():
            combined_features = model.feat(data)
        extracted_features.append(combined_features.cpu().numpy())

        with torch.no_grad():
            pred = model.classifier(combined_features)
            probs = torch.sigmoid(pred).squeeze()

        all_prob.append(probs.detach().cpu())
        all_target.append(data.y.float().squeeze().detach().cpu())
        labels.append(data.y.cpu().numpy())

    all_targets_tensor = torch.cat(all_target, dim=0)
    all_probs_tensor = torch.cat(all_prob, dim=0)

    threshold, _ = find_best_threshold_by_f_beta(all_targets_tensor, all_probs_tensor, num_threshold=100)
    metrics = calculate_metrics(y_true=all_targets_tensor, y_scores=all_probs_tensor, threshold=threshold)

    if visualize:
        extracted_features = torch.tensor(np.concatenate(extracted_features, axis=0))
        labels = np.concatenate(labels, axis=0)

        pca = PCA(n_components=2)
        extracted_pca = pca.fit_transform(extracted_features)

        fig, axs = plt.subplots(1, 1, figsize=(6, 5))
        scatter = axs.scatter(extracted_pca[:,0], extracted_pca[:,1], c=labels, cmap='coolwarm', alpha=0.6, s=1)
        axs.set_title("Fused Features PCA")
        fig.colorbar(scatter, ax=axs)

        plt.tight_layout()
        plt.savefig(f"{Dset_name}_fused_pca_vis.png", dpi=300)
        plt.close()

    return metrics

def main():
    parse=argparse.ArgumentParser(description='PPI Val')
    parse.add_argument('--data', required=True, type=str)
    parse.add_argument('--Dset_name', required=True, type=str)
    args = parse.parse_args()
    data_loader = DataLoader(device=device, multimodal_data_dir=eval(args.data))
    all_proteins = DataLoader.load_data(data_loader)
    dat_info_sample = data_loader.prepare_sample(all_proteins[0])
    
    modal_dims_info = DataLoader.dat_ifo(dat_info_sample)
    
    modal_cfg = []
    if hasattr(dat_info_sample, 'modal_names_list'):
        for modal_name in dat_info_sample.modal_names_list:
            cfg_entry = {
                'name': modal_name,
                'in_channels': modal_dims_info.get(f'{modal_name}_in_channels', 0),
                'pe_dim': modal_dims_info.get(f'{modal_name}_pe_dim', config.PE_DIM),
            }
            modal_cfg.append(cfg_entry)
    seed = config.SEED
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_class = DualStreamPPI
    model = model_class(
        modal_cfg=modal_cfg,
        out_channels=config.OUT_CHANNELS
    ).to(device)

    best_model_path = os.path.join(config.PRE_MODEL, f'Train.pth')
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    metrics = A(model, all_proteins, data_loader, visualize=True, Dset_name=args.Dset_name)
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

    save_metrics_to_txt(metrics, args.Dset_name)
if __name__ == '__main__':
    main()
