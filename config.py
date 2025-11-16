import torch

DEVICE = torch.device('cuda:0')
SEED = 602025

DATA_DIR = ['/../gz-data/Train/Train2066_ESM.pkl', '/../gz-data/Train/Train2066_Prot.pkl', '/../gz-data/Train/Train2066_aaindex1.pkl', '/../gz-data/Train/Train2066_edge.pkl', '/../gz-data/Train/Train2066_label.pkl']
VAL1 = ['/../gz-data/Test/Test60_ESM.pkl', '/../gz-data/Test/Test60_Prot.pkl', '/../gz-data/Test/Test60_aaindex1.pkl', '/../gz-data/Test/Test60_edge.pkl', '/../gz-data/Test/Test60_label.pkl']
VAL2 = ['/../gz-data/Test/Test315_ESM.pkl', '/../gz-data/Test/Test315_Prot.pkl', '/../gz-data/Test/Test315_aaindex1.pkl', '/../gz-data/Test/Test315_edge.pkl', '/../gz-data/Test/Test315_label.pkl']
VAL3 = ['/../gz-data/Test/UBtest31_ESM.pkl', '/../gz-data/Test/UBtest31_Prot.pkl', '/../gz-data/Test/UBtest31_aaindex1.pkl', '/../gz-data/Test/UBtest31_edge.pkl', '/../gz-data/Test/UBtest31_label.pkl']
VAL4 = ['/../gz-data/Test/Train362_ESM.pkl', '/../gz-data/Test/Train362_Prot.pkl', '/../gz-data/Test/Train362_aaindex1.pkl', '/../gz-data/Test/Train362_edge.pkl', '/../gz-data/Test/Train362_label.pkl']

NUM_LAYER = 4
HEADS = 8
DROPOUT = 0.4
gcn_hid_dim = 256

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 5e-3
PATIENCE = 8

A = 0.36     #wight of FN
B = 0.97      #weght of FP
BCE_WEIGHT = 0.3
FOCAL_WEIGHT = 0
Tversky_WEIGHT = 0.7

PRE_MODEL = '/../gz-data/TRAINING_OUTPUT/Saved_model'
TUNING_MODEL = '/../gz-data/TRAINING_OUTPUT/Tuning_model'
PLOT_DIR = '/../gz-data/TRAINING_OUTPUT/plots'

T_MAX = EPOCHS
ETA_MIN = 1e-6
