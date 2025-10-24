import torch

DEVICE = torch.device('cuda:0')
SEED = 649737
# There are 306228 binding stes and 1649339 non-binding sites
DATA_DIR = r'/../gz-data/Train/'
# There are 10904 binding stes and 54932 non-binding sites
TUNING_DATA_PATH = r'/../gz-data/Tune'
# There are 1962 binding stes and 10098 non-binding sites
VAL_DATA_PATH = r'/../gz-data/Test'

PE_DIM = 32

FEAT_GNN_HID_DIM = 256
GEO_GNN_HID_DIM = 64
EDGE_DIM = 1

NUM_LAYER = 4
Dual_FUSE_DIM = 128
OUT_CHANNELS = 1
HEADS = 4      # from 3 to 2 time:2025-9-23 22:18
DROPOUT = 0.2

EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 4e-3    # from 4e-4 to 1e-3time: 2025-9-23 22:26
WEIGHT_DECAY = 4e-4
POS_WEIGHT = 0.2       # Change from 1 to 0.5 time 2025-9-23 12:41
GRAD_NORM = 0.5
PATIENCE = 15

ALPHA = 0.3
BETA=0.97
B_WEIGHT = 0.36
T_WEIGHT = 0.97

PRE_MODEL = '/../gz-data/TRAINING_OUTPUT/Saved_model'
TUNING_MODEL = '/../gz-data/TRAINING_OUTPUT/Tuning_model'
PLOT_DIR = '/../gz-data/TRAINING_OUTPUT/plots'

SCHEDULER_T_MAX = EPOCHS // 2
SCHEDULER_ETA_MIN = 1e-6
