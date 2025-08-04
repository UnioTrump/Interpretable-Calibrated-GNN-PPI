import torch

# --- GENERAL ---
PROJECT_NAME = 'GASPPI_Mamba'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

# --- PATHS ---
PLOT_DIR = './plots'
MODEL_DIR = './models'

# --- FEATURE CONFIG ---
ATOM_DISTANCE_THRESHOLD = 4.5
RESIDUE_DISTANCE_THRESHOLD = 6.0

# --- MODEL HYPERPARAMETERS (shared for both stages) ---
HIDDEN_DIM = 256
OUT_CHANNELS = 1
NUM_ATOM_LAYERS = 2
NUM_RESIDUE_LAYERS = 4
HEADS = 4
PE_DIM = 16
DROPOUT = 0.2

# --- MAMBA CONFIG (shared for both stages) ---
MAMBA_D_STATE = 16
MAMBA_D_CONV = 4
MAMBA_EXPAND = 2

# --- PRE-TRAINING CONFIG ---
PRETRAIN = {
    'DATA_PATH': './data/pretrain_dataset.pkl',  # <-- 请将这里替换为你的3960个蛋白质的数据集路径
    'EPOCHS': 150,
    'BATCH_SIZE': 32,
    'LEARNING_RATE': 1e-4,
    'WEIGHT_DECAY': 0.01,
    'POS_WEIGHT': 1,
    'GRAD_NORM': 1.0,
    'PATIENCE': 20,
    'SCHEDULER_T_MAX': 100,
    'SCHEDULER_ETA_MIN': 1e-6,
    'MODEL_SAVE_PATH': f'{MODEL_DIR}/{PROJECT_NAME}_pretrained.pth'
}

# --- FINE-TUNING CONFIG ---
FINETUNE = {
    'PRETRAINED_PATH': PRETRAIN['MODEL_SAVE_PATH'],  # Automatically use the saved pre-trained model
    'DATA_PATH': './data/Train_334.pkl', # <-- 这里使用你的334个蛋白质的数据集
    'VAL_DATA_PATH': './data/Test_60.pkl', # <-- 用于微调时的验证集
    'EPOCHS': 80,
    'BATCH_SIZE': 16,
    'LEARNING_RATE': 1e-5,  #  <-- 使用更小的学习率
    'WEIGHT_DECAY': 0.01,
    'POS_WEIGHT': 1,
    'GRAD_NORM': 1.0,
    'PATIENCE': 30,
    'SCHEDULER_T_MAX': 50,
    'SCHEDULER_ETA_MIN': 1e-6,
    'MODEL_SAVE_PATH': f'{MODEL_DIR}/{PROJECT_NAME}_finetuned_best.pth'
}
