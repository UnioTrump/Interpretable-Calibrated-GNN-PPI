import torch

# --- GENERAL ---
PROJECT_NAME = 'GASPPI_Mamba'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

# --- DATA ---
DATA_PATH = '/../gz-data/features/Final/Train/Train_334.pkl' 
VAL_DATA_PATH = '/../gz-data/features/Final/Test/Test_60.pkl' 
PLOT_DIR = './plots'
MODEL_DIR = './models'

# --- MODEL HYPERPARAMETERS ---
HIDDEN_DIM = 256
OUT_CHANNELS = 1
NUM_ATOM_LAYERS = 2
NUM_RESIDUE_LAYERS = 4
NUM_SEQ_LAYERS = 2
HEADS = 4
PE_DIM = 16
DROPOUT = 0.2
VOCAB_SIZE = 21

# --- MAMBA CONFIG ---
MAMBA_D_STATE = 16
MAMBA_D_CONV = 4
MAMBA_EXPAND = 2

# --- TRAINING ---
EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
POS_WEIGHT = 1
GRAD_NORM = 1e-2
PATIENCE = 8

# --- SCHEDULER ---
SCHEDULER_T_MAX = EPOCHS // 2
SCHEDULER_ETA_MIN = 1e-6
