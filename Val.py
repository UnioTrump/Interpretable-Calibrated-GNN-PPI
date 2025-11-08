import torch
import os
from utils import calculate_metrics, find_best_threshold_by_f_beta, save_metrics_to_txt
from model import PPI
import config
from tqdm import tqdm
from Data import Dataloader
import numpy as np
import argparse

device = config.DEVICE

@torch.no_grad()
def A(model, val_proteins, data_loader):
    model.eval()
    all_prob, all_target = [], []
    extracted_features, labels = [], []

    for sample_data in tqdm(val_proteins, desc='Testing'):
        data = data_loader.prepare_sample(sample_data)

        with torch.no_grad():
            out = model(ax=data.aa, bx=data.esm, cx=data.prot, adj=data.adj)
        extracted_features.append(out.cpu().numpy())

        with torch.no_grad():
            probs = torch.sigmoid(out).squeeze()

        all_prob.append(probs.detach().cpu())
        all_target.append(data.y.float().squeeze().detach().cpu())
        labels.append(data.y.cpu().numpy())

    all_targets_tensor = torch.cat(all_target, dim=0)
    all_probs_tensor = torch.cat(all_prob, dim=0)

    threshold, _ = find_best_threshold_by_f_beta(all_targets_tensor, all_probs_tensor, num_threshold=100)
    metrics = calculate_metrics(y_true=all_targets_tensor, y_scores=all_probs_tensor, threshold=threshold)

    return metrics

def main():
    parse=argparse.ArgumentParser(description='PPI Val')
    parse.add_argument('--data', required=True, type=str)
    parse.add_argument('--Dset_name', required=True, type=str)
    args = parse.parse_args()
    data_loader = Dataloader()
    all_proteins = Dataloader.load_data(eval(args.data))

    seed = config.SEED
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = PPI(hid_dim=config.gcn_hid_dim, heads=config.HEADS, dropout=config.DROPOUT, bi=config.bi).to(device)

    best_model_path = os.path.join(config.PRE_MODEL, f'Train.pth')
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    model.eval()

    metrics = A(model, all_proteins, data_loader)
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
