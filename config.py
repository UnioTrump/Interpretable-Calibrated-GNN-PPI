import torch

# General
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
PROJECT_NAME = "EnhancedSpectralPPI_v3"

# Data
DATA_PATH = r'/gz-data/Train'
VAL_DATA_PATH = r'/gz-data/Test/TestUB25.pkl'
PE_DIM = 16
GAUSSIAN_SIGMA = 1.0

# Model
ATOM_HIDDEN_DIMS = [64, 128]
RESIDUE_HIDDEN_DIMS = [256, 256, 128]
GEO_HIDDEN_DIMS = [64, 128]
FUSION_HIDDEN_DIM = 128
OUT_CHANNELS = 1
HEADS = 4
DROPOUT = 0.5

# Enhanced Spectral Attention Parameters
MAX_EIGENVECTORS = 32  # 最大特征向量数量
SPECTRAL_ENHANCED = True  # 是否启用谱增强
TASK_TYPE = "general"  # 任务类型: "physical_interaction", "functional_association", "pathway_coregulation", "general"
ENHANCEMENT_LEVEL = 2  # 增强级别: 0=原始, 1=部分增强, 2=完全增强

# Multi-Scale Parameters
NUM_SCALES = 3  # 多尺度数量 (低频、中频、高频)
FREQ_THRESHOLD_LOW = 0.33  # 低频阈值
FREQ_THRESHOLD_HIGH = 0.67  # 高频阈值

# Adaptive Frequency Weighting (can be overridden by TASK_TYPE)
FREQ_WEIGHT_HIGH = 0.33  # 高频权重
FREQ_WEIGHT_MID = 0.34   # 中频权重  
FREQ_WEIGHT_LOW = 0.33   # 低频权重

# Training
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 4e-4
POS_WEIGHT = 1
GRAD_NORM = 1.0
PATIENCE = 8
MODEL_DIR = './saved_models'
PLOT_DIR = './plots'
SCHEDULER_T_MAX = EPOCHS // 2
SCHEDULER_ETA_MIN = 1e-6

# Experimental Settings
USE_ENHANCED_DEMO = False  # 是否使用增强版demo
COMPARE_MODELS = True     # 是否进行模型对比
SAVE_SPECTRAL_INFO = True # 是否保存谱分析信息

# Performance Monitoring
LOG_SPECTRAL_STATS = True  # 记录谱统计信息
PLOT_ATTENTION_WEIGHTS = False  # 绘制注意力权重（计算密集）
SAVE_ATTENTION_MAPS = False     # 保存注意力图（存储密集）
