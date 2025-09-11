import torch

# General
DEVICE = torch.device('cuda:0')
SEED = [649737, 395408, 252356, 343053, 743746]
PROJECT_NAME = "DualStreamPPI_v2"

# Data
'''
In pretrain data, there are 977 proteins with 244828 residues.
There are 46353 residue with pos label, and there are 198475 residue with neg label
'''
DATA_PATH = r'/../gz-data/features/Pretrain'
'''
In the Tuning data, there are 330 proteins with 67069 residues
There are 10714 residues with pos label and 56025 residues with neg label
'''
TUNING_DATA_PATH = r'/../gz-data/features/Final/Tuning'
'''
In the Test data, there are 60 proteins with 10677 residues
There are 1929 residues with pos label and 8688 residues with neg label
'''
VAL_DATA_PATH = r'/../gz-data/features/Final/Test'
PE_DIM = 16
GAUSSIAN_SIGMA = 1
FOURIER_THRESHOLD = 1.0

# Model
ATOM_HIDDEN_DIMS = [256, 512]
RESIDUE_HIDDEN_DIMS = [1024, 512]
GEO_HIDDEN_DIMS = [32, 64]
FUSION_HIDDEN_DIM = 128
OUT_CHANNELS = 1
HEADS = 4
DROPOUT = 0.2

# Training
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 4e-4
POS_WEIGHT = 1
GRAD_NORM = 0.5
PATIENCE = 8
PRE_MODEL = '/../gz-data/TRAINING_OUTPUT/Saved_model'
TUNING_MODEL = '/../gz-data/TRAINING_OUTPUT/Tuning_model'
PLOT_DIR = '/../gz-data/TRAINING_OUTPUT/plots'
SCHEDULER_T_MAX = EPOCHS
SCHEDULER_ETA_MIN = 1e-6
