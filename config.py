import torch

# General
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
PROJECT_NAME = "DualStreamPPI_v2"

# Data
DATA_PATH = r'/gz-data/Train/Train335.pkl'
VAL_DATA_PATH = r'/gz-data/Test/Test70.pkl'
PE_DIM = 16
GAUSSIAN_SIGMA = 1.0

# Model
ATOM_HIDDEN_DIMS = [64, 128]
RESIDUE_HIDDEN_DIMS = [256, 256, 128]
GEO_HIDDEN_DIMS = [64, 128]
FUSION_HIDDEN_DIM = 128
OUT_CHANNELS = 1
HEADS = 4
DROPOUT = 0.5

# Training
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
POS_WEIGHT = 1
GRAD_NORM = 1.0
PATIENCE = 8
MODEL_DIR = './saved_models'
PLOT_DIR = './plots'
SCHEDULER_T_MAX = EPOCHS // 2
SCHEDULER_ETA_MIN = 1e-6
