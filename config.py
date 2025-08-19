import torch

# General
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
PROJECT_NAME = "DualStreamPPI_v2"

# Data
'''
In pretrain data, there are 977 proteins with 244828 residues.
There are 46353 residue with pos label, and there are 198475 residue with neg label
'''
DATA_PATH = r'/../gz-data/features/Final/Pretrain'
TUNING_DATA_PATH = r'/../gz-data/features/Final/Train'
PE_DIM = 16
GAUSSIAN_SIGMA = 0.1

# Model
ATOM_HIDDEN_DIMS = [128, 64]
RESIDUE_HIDDEN_DIMS = [256, 128]
GEO_HIDDEN_DIMS = [128, 64]
FUSION_HIDDEN_DIM = 128
OUT_CHANNELS = 1
HEADS = 4
DROPOUT = 0.7

# Training
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 4e-3
WEIGHT_DECAY = 1e-2
POS_WEIGHT = 1
GRAD_NORM = 1.0
PATIENCE = 8
PRE_MODEL = '/../gz-data/TRAINING_OUTPUT/Saved_model'
TUNING_MODEL = '/../gz-data/TRAINING_OUTPUT/Tuning_model'
PLOT_DIR = '/../gz-data/TRAINING_OUTPUT/plots'
SCHEDULER_T_MAX = EPOCHS // 2
SCHEDULER_ETA_MIN = 1e-6
