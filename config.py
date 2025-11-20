import torch

DEVICE = torch.device('cuda:0')
SEED = 2025

DATA_DIR = '../gz-data/Train2066/'
VAL1 = '../gz-data/Test60/'
VAL2 = '../gz-data/Test315/'
VAL3 = '../gz-data/Train362/'

NUM_LAYER = 2
HEADS = 8
DROPOUT = 0.3
gcn_hid_dim = 512

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 5e-3
PATIENCE = 8
K_FOLDS = 5

A = 0.4     #wight of FN
B = 0.8      #weght of FP
BCE_WEIGHT = 0.3
FOCAL_WEIGHT = 0
Tversky_WEIGHT = 0.7

PRE_MODEL = '/../gz-data/TRAINING_OUTPUT/Saved_model'
TUNING_MODEL = '/../gz-data/TRAINING_OUTPUT/Tuning_model'
PLOT_DIR = '/../gz-data/TRAINING_OUTPUT/plots'

T_MAX = EPOCHS
ETA_MIN = 1e-6
