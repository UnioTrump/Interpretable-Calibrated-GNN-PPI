from tqdm import tqdm
import torch
import os
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import WeightedCrossEntropy, calculate_metrics, find_best_threshold_by_f_beta, plot_loss_curves
from GASPPI import DualStreamPPI
import config
from data_utils import DataLoader
device=config.DEVICE
def run_train():
    data_loader = DataLoader(device=device)
    all_proteins = data_loader.load_data(config.DATA_PATH)

