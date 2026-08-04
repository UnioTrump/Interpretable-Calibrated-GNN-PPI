import torch

DEVICE = torch.device('cuda:0')
SEED = 2025

DATA_DIR = '../gz-data/Train1958/'
VAL1 = '../gz-data/Test_60/'
VAL2 = '../gz-data/Test_315/'
VAL3 = '../gz-data/DSet_72/'
VAL4 = '../gz-data/DSet_164/'
VAL5 = '../gz-data/DSet_186/'

NUM_LAYER = 4
HEADS = 8
DROPOUT = 0.2
gcn_hid_dim = 128

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 4e-4
PATIENCE = 8
K_FOLDS = 5

A = 0.3     #wight of FP
B = 0.97      #weight of FN
POS_WEIGHT = torch.tensor(0.1)
BCE_WEIGHT = 0.3
FOCAL_WEIGHT = 0
Tversky_WEIGHT = 0.7

F_BETA = 0.7

PRE_MODEL = '../gz-data/Pre_model'
TUNING_MODEL = '../gz-data/Tuning_model'
PLOT_DIR = '../gz-data/plots'

T_MAX = EPOCHS
ETA_MIN = 1e-6
