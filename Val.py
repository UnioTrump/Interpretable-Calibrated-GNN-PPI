import torch
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import calculate_metrics, find_best_threshold_by_f_beta
from GASPPI import DualStreamPPI
import config
from tqdm import tqdm
from data_utils import DataLoader
import numpy as np

device = config.DEVICE

@torch.no_grad()
def A(models, val_proteins, data_loader, visualize=True):
    all_prob, all_target = [], []
    raw_features, extracted_features, labels = [], [], []

    for val_p_idx in tqdm(val_proteins, desc='Testing'):
        data = data_loader.prepare_sample(val_p_idx)

        raw_feat = data.seq_x.cpu().numpy()
        raw_features.append(raw_feat)

        model_features = []
        for model in models:
            with torch.no_grad():
                out = model.feat(data)
                model_features.append(out)

        features_tensor = torch.stack(model_features, dim=0)
        combined_features = features_tensor.mean(dim=0)
        extracted_features.append(combined_features.cpu().numpy())

        with torch.no_grad():
            pred = models[0].MLP(combined_features)
            probs = torch.sigmoid(pred).squeeze()

        all_prob.append(probs.detach().cpu())
        all_target.append(data.y.float().squeeze().detach().cpu())
        labels.append(data.y.cpu().numpy())

    all_targets_tensor = torch.cat(all_target, dim=0)
    all_probs_tensor = torch.cat(all_prob, dim=0)

    threshold, _ = find_best_threshold_by_f_beta(all_targets_tensor, all_probs_tensor, num_threshold=100)
    metrics = calculate_metrics(y_true=all_targets_tensor, y_scores=all_probs_tensor, threshold=threshold)

    # --------- PCA可视化部分 ---------
    if visualize:
        raw_features = torch.tensor(np.concatenate(raw_features, axis=0))
        extracted_features = torch.tensor(np.concatenate(extracted_features, axis=0))
        labels = np.concatenate(labels, axis=0)

        # PCA降维到2D
        pca = PCA(n_components=2)
        raw_pca = pca.fit_transform(raw_features)
        extracted_pca = pca.fit_transform(extracted_features)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        scatter1 = axs[0].scatter(raw_pca[:,0], raw_pca[:,1], c=labels, cmap='coolwarm', alpha=0.6, s=2)
        axs[0].set_title("Raw Features PCA")
        fig.colorbar(scatter1, ax=axs[0])

        scatter2 = axs[1].scatter(extracted_pca[:,0], extracted_pca[:,1], c=labels, cmap='coolwarm', alpha=0.6, s=2)
        axs[1].set_title("Extracted Features PCA")
        fig.colorbar(scatter2, ax=axs[1])

        plt.tight_layout()
        plt.savefig("pca_vis.png", dpi=300)
        plt.close()

    return metrics

def main():
    models = []
    data_loader = DataLoader(
        device=device,
        multimodal_data_dir=config.VAL_DATA_PATH
    )
    all_proteins = DataLoader.load_data(data_loader)
    
    if all_proteins:
        sample_data_for_info = data_loader.prepare_sample(all_proteins[0])
    else:
        raise ValueError("No data loaded for validation. Please check data paths.")
    
    dat_info = DataLoader.data_ifo(sample_data_for_info)
    sequence_in_channels = dat_info['sequence_in_channels']
    modal2_in_channels = dat_info.get('modal2_in_channels', None)
    modal3_in_channels = dat_info.get('modal3_in_channels', None)
    modal2_pe_dim = dat_info.get('modal2_pe_dim', None)
    modal3_pe_dim = dat_info.get('modal3_pe_dim', None)

    for index, seed in enumerate(config.SEED):
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)

        model_class = DualStreamPPI
        model = model_class(
            in_channels=sequence_in_channels,
            pe_dim=config.PE_DIM,
            out_channels=config.OUT_CHANNELS,
            modal2_in_channels=modal2_in_channels,
            modal3_in_channels=modal3_in_channels,
            modal2_pe_dim=modal2_pe_dim,
            modal3_pe_dim=modal3_pe_dim
        ).to(device)

        best_model_path = os.path.join(config.TUNING_MODEL, f'{index}_best_model.pth')
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        models.append(model)

    metrics = A(models, all_proteins, data_loader, visualize=True)
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
