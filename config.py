import torch

DEVICE = torch.device('cuda:0')
SEED = 2025

DATA_DIR = '../data/Train_1960/'
VAL1 = '../data/Test_60/'
VAL2 = '../data/Test_315/'
VAL3 = '../data/DSet_72/'
VAL4 = '../data/DSet_164/'
VAL5 = '../data/DSet_186/'

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
BCE_WEIGHT = 0.3
FOCAL_WEIGHT = 0
Tversky_WEIGHT = 0.7

F_BETA = 0.7

PRE_MODEL = '/../data/Pre_model'
TUNING_MODEL = '/../data/Tuning_model'
PLOT_DIR = '/../data/plots'

T_MAX = EPOCHS
ETA_MIN = 1e-6
