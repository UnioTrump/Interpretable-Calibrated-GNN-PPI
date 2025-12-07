import torch

DEVICE = torch.device('cuda:0')
SEED = 2025

DATA_DIR = '../gz-data/Train2066/'
VAL1 = '../gz-data/Test60/'
VAL2 = '../gz-data/Test315/'
VAL3 = '../gz-data/Train362/'
VAL4 = '../gz-data/UBtest31/'

NUM_LAYER = 4
HEADS = 8
DROPOUT = 0.2
gcn_hid_dim = 128

EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 4e-4
PATIENCE = 8
K_FOLDS = 5

A = 0.3     #wight of FP
B = 0.8      #weght of FN
BCE_WEIGHT = 0.3
FOCAL_WEIGHT = 0
Tversky_WEIGHT = 0.7

F_BETA = 0.7

PRE_MODEL = '/../gz-data/TRAINING_OUTPUT/Saved_model'
TUNING_MODEL = '/../gz-data/TRAINING_OUTPUT/Tuning_model'
PLOT_DIR = '/../gz-data/TRAINING_OUTPUT/plots'

T_MAX = EPOCHS
ETA_MIN = 1e-6
