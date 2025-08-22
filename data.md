# 晶体Transformer模型框架（4任务多任务学习）

本文详细介绍了晶体Transformer（CT）模型的架构、数据准备和训练过程，该模型专门适用于4任务多任务学习。CT模型利用基于Transformer的编码器，根据晶体结构同时预测多种材料属性。

## 1. 模型架构 (`ct/model_mt_4.py`)

`ct/model_mt_4.py` 中的 `CrystalTransformer` 类旨在处理原子特征及其3D坐标，以预测四种不同的材料属性。

**输入：**
模型接收三个主要输入：
*   `atom`：原子特征。
*   `coords`：原子坐标（3D）。
*   `mask`：用于处理批次中可变长度结构的布尔掩码，指示填充元素。

**嵌入层：**
原子和坐标数据的初始转换由线性嵌入层完成：
*   **原子嵌入：** 将100维原子特征向量映射到 `feature_size / 2`。
    $\text{Atom\_embedding} = \text{Linear\_atom}(\text{atom})$
*   **坐标嵌入：** 将3D坐标映射到 `feature_size / 2`。
    $\text{Coord\_embedding} = \text{Linear\_coord}(\text{coords})$

**Transformer输入拼接：**
嵌入后的原子特征和坐标被拼接起来，形成Transformer编码器的源输入 `src`：
$\text{src} = [\text{Atom\_embedding} ; \text{Coord\_embedding}]$
`src` 张量的维度将为 `feature_size`。

**Transformer编码器：**
`src` 输入由Transformer编码器处理，该编码器由 `num_layers` 个 `TransformerEncoderLayer` 组成。每个层通常包括：
*   **多头自注意力机制：** 使模型能够通过不同地加权输入元素来捕获原子间的关系。
    $\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}) \mathbf{V}$
    其中 $\mathbf{Q}$、$\mathbf{K}$、$\mathbf{V}$ 是从输入派生的查询、键和值矩阵，$\text{d}_k$ 是键的维度。
*   **前馈网络：** 对每个位置独立应用的逐位置全连接前馈网络，用于非线性转换。
*   **残差连接和层归一化：** 在自注意力和前馈层之后应用，以实现稳定高效的训练。

Transformer编码器的输出为：
$\text{output} = \text{TransformerEncoder}(\text{src}, \text{src\_key\_padding\_mask}=\text{mask})$

**多任务输出层：**
Transformer编码器中对应 `[CLS]` 标记（或第一个标记，`output[:, 0, :]`）的表示通过四个独立的输出分支，用于预测四种不同的材料属性：
*   **输出 1：** $\text{Output}_1 = \text{Linear}_{1b}(\text{ReLU}(\text{Linear}_{1a}(\text{output}[:, 0, :]))) $
*   **输出 2：** $\text{Output}_2 = \text{Linear}_{2b}(\text{ReLU}(\text{Linear}_{2a}(\text{output}[:, 0, :]))) $
*   **输出 3：** $\text{Output}_3 = \text{Linear}_{3b}(\text{ReLU}(\text{Linear}_{3a}(\text{output}[:, 0, :]))) $
*   **输出 4：** $\text{Output}_4 = \text{Linear}_{4b}(\text{ReLU}(\text{Linear}_{4a}(\text{output}[:, 0, :]))) $
每个 $\text{Linear}_{X_a}$ 将 `feature_size` 映射到 `feature_size`，每个 $\text{Linear}_{X_b}$ 将 `feature_size` 映射到 1，为每个任务生成一个标量预测。

## 2. 数据准备 (`ct/data_mt_4.py`)

`ct/data_mt_4.py` 文件负责为4任务多任务学习处理晶体结构数据的加载、预处理和批处理。

### 2.1. CIFData 类

`CIFData` 类（一个 PyTorch `Dataset`）管理单个晶体结构的加载和初始处理。

*   **初始化：** 需要 `root_dir`（包含 CIF 文件和 `id_prop.csv`）和一个 `id_prop.csv` 文件，该文件结构应包含每个条目的四个目标属性。它还使用 `atom_init.json` 进行原子特征初始化。
*   **`__getitem__(self, idx)`：**
    *   从 `id_prop_data` 中读取 `cif_id` 和四个 `target` 属性（`target1`、`target2`、`target3`、`target4`）。
    *   使用 `pymatgen` 从 `cif_id + '.cif'` 加载晶体结构。
    *   生成 `atom_fea`（原子特征向量）和 `atom_coords`（原子坐标）。
    *   通过 `_augment_coordinates()` 以50%的概率对 `atom_coords` 应用坐标增强（随机平移、旋转和反射）。
    *   返回 `(atom_fea, atom_coords)`、`target1`、`target2`、`target3`、`target4` 和 `cif_id`。

### 2.2. 整理函数（`collate_pool` 和 `collate_pool_train`）

这些函数负责将单个样本分组为批次，用于训练和验证。

*   **填充：** 原子特征和坐标用零填充，以确保批次中张量大小一致，最大可达 `N_max`（当前批次中的最大原子数）。
*   **掩码创建：** 为Transformer编码器生成一个布尔 `mask`，以忽略填充元素。`True` 表示填充（被忽略）元素。
    $\text{mask}[i] = 1 \text{ 如果原子真实存在, } 0 \text{ 如果是填充}$
    然后将掩码反转，用作 `src_key_padding_mask`。
*   **数据增强（在 `collate_pool_train` 中）：**
    *   以50%的概率向 `atom_fea` 和 `atom_coords` 添加高斯噪声。
    *   以50%的概率应用坐标增强（平移、旋转、反射），类似于 `__getitem__`。

## 3. 训练和验证过程 (`train/main_mt_4.py`)

`train/main_mt_4.py` 脚本协调4任务多任务 `CrystalTransformer` 模型的端到端训练和验证。

### 3.1. 数据加载和归一化

*   脚本使用 `ct/data_mt_4.py` 中的 `get_train_val_test_loader` 创建 `train_loader`、`val_loader` 和 `test_loader`。
*   创建四个 `Normalizer` 实例（每个目标属性一个）来归一化目标值。这对于稳定训练至关重要，尤其是在目标属性具有不同尺度时。
    $\text{target\normed}_j = \frac{\text{target}_j - \mu_j}{\sigma_j}$
    其中 $\mu_j$ 和 $\sigma_j$ 分别是第 $j$ 个目标属性的均值和标准差。

### 3.2. 模型、损失和优化器

*   **模型实例化：** 使用指定的 `feature_size`、`num_layers`、`num_heads` 和 `dim_feedforward` 初始化 `CrystalTransformer` 模型。
*   **损失函数：** 均方误差（MSE）用作每个任务的判据。
    $\text{MSE}_j = \frac{1}{N} \sum_{i=1}^{N} (\text{output}_{i,j} - \text{target\_normed}_{i,j})^2$
*   **总损失：** 多任务学习的总损失是单个任务损失的总和。
    $\text{Loss}_{total} = \text{Loss}_1 + \text{Loss}_2 + \text{Loss}_3 + \text{Loss}_4$
*   **优化器：** 使用SGD或Adam优化器。如果 `load_embedding_only` 为真，则可以对原子嵌入参数应用单独的（通常较小的）学习率。

### 3.3. 训练循环（`train` 函数）

*   **前向传播：** 对于每个批次，模型将 `atom`、`coords` 和 `mask` 作为输入，并生成四个输出（`output1`、`output2`、`output3`、`output4`）。
*   **损失计算：** 每个输出与使用MSE的相应归一化目标进行比较。将单个损失相加得到 `Loss_{total}`。
*   **反向传播和优化：** `Loss_{total}.backward()` 计算梯度，`optimizer.step()` 更新模型参数。
*   **评估指标：** 为每个任务计算平均绝对误差（MAE）和R平方（$R^2$），以监控性能。
    *   **MAE：** $\text{MAE}_j = \frac{1}{N} \sum_{i=1}^{N} |\text{denorm}(\text{output}_{i,j}) - \text{target}_{i,j}|$
    *   **R平方（$R^2$）：** $R^2_j = 1 - \frac{\sum_{i} (\text{target}_{i,j} - \text{denorm}(\text{output}_{i,j}))^2}{\sum_{i} (\text{target}_{i,j} - \bar{\text{target}}_j)^2}$

### 3.4. 验证循环（`validate` 函数）

*   `validate` 函数评估模型在验证（或测试）集上的性能。
*   它计算四个任务的MAE和$R^2$，而不更新模型权重。
*   根据总体MAE误差保存最佳模型检查点。
