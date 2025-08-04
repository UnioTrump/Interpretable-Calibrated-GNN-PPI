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

# --- MODEL HYPERPARAMETERS (H-GNN-Mamba Architecture) ---
# Core dimensions
HIDDEN_DIM = 128  # Unified hidden dimension for all embeddings
OUT_CHANNELS = 1  # For binary classification


# Hierarchical settings
NUM_ATOM_LAYERS = 2 # Number of UnifiedEncoderBlock for the atom encoder
NUM_RESIDUE_LAYERS = 4 # Number of UnifiedEncoderBlock for the residue encoder

# Interaction Block settings (now part of UnifiedEncoderBlock)
HEADS = 4  # Number of attention heads in GNN layers

# Mamba-specific settings
MAMBA_D_STATE = 16
MAMBA_D_CONV = 4
MAMBA_EXPAND = 2

# Other shared parameters
PE_DIM = 16  # Dimensionality of Laplacian Positional Encodings
DROPOUT = 0.2

# --- TRAINING ---
EPOCHS = 200
BATCH_SIZE = 16 # Process one protein graph at a time
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
POS_WEIGHT = 3.0  # For weighted cross-entropy loss
GRAD_NORM = 1.0  # Gradient clipping norm
PATIENCE = 30 # Patience for early stopping

# --- SCHEDULER ---
SCHEDULER_T_MAX = 200
SCHEDULER_ETA_MIN = 1e-6
