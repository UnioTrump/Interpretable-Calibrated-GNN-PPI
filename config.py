import torch

DEVICE = torch.device('cuda:0')
SEED = 2025
# There are 306228 binding stes and 1649339 non-binding sites
DATA_DIR = ['/../gz-data/preTrain/Trainset7596_ESMC.pkl', '/../gz-data/preTrain/Trainset7596_ProtT5.pkl', '/../gz-data/preTrain/Trainset7596.pkl', '/../gz-data/preTrain/Trainset7596_attn.pkl', '/../gz-data/preTrain/Trainset7596_label.pkl']
# DATA_DIR = ['/../gz-data/Train/Train362_ESMC.pkl', '/../gz-data/Train/Train362_ProtT5.pkl', '/../gz-data/Train/Train362_ab.pkl', '/../gz-data/Train/Train362_attn.pkl', '/../gz-data/Train/Train362_label.pkl']
# There are 10904 binding stes and 54932 non-binding sites
# TUNING_DATA_PATH = r'/../gz-data/Tune'
# There are 1962 binding stes and 10098 non-binding sites
VAL_DATA_PATH = ['/../gz-data/Test/Test60_ESMC.pkl', '/../gz-data/Test/Test60_Prot.pkl', '/../gz-data/Test/Test60_ab.pkl', '/../gz-data/Test/Test60_attn.pkl', '/../gz-data/Test/Test60_label.pkl']

NUM_LAYER = 5
DROPOUT = 0.1

EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 4e-3
WEIGHT_DECAY = 4e-4
POS_WEIGHT = 1
GRAD_NORM = 0.5
PATIENCE = 15

ALPHA = 0.7
LAMDA = 1.5

A = 0.3     #wight of FN
B=0.97      #weght of FP
B_WEIGHT = 0.36     # weight of BCELoss
T_WEIGHT = 0.97     # weight of Tversky loss

PRE_MODEL = '/../gz-data/TRAINING_OUTPUT/Saved_model'
TUNING_MODEL = '/../gz-data/TRAINING_OUTPUT/Tuning_model'
PLOT_DIR = '/../gz-data/TRAINING_OUTPUT/plots'

SCHEDULER_T_MAX = EPOCHS // 2
SCHEDULER_ETA_MIN = 1e-6
