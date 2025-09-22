# DualStreamPPI - 多模态数据预处理

本文档描述了 `DualStreamPPI` 模型的数据预处理流程，将原始大模型语义嵌入转换为图结构数据。主要步骤在 `preprocess.py` 和 `data_utils/data_utils.py` 中实现。

## 1. 原始数据

初始数据来源于 `esmc_600m`, `ProtT5`, `ProtBERT` 等大模型提取的蛋白质语义嵌入，以 `.pkl` 文件形式存储，每个文件包含一个模态的所有蛋白质样本列表。

## 2. 数据预处理流程 (`preprocess.py`)

`preprocess.py` 脚本将原始语义嵌入转换为图神经网络格式。

### 2.1. `process_single_protein` 函数

此函数处理单个蛋白质样本的预处理，生成核心数据：`r_node` (节点特征), `residue_adj_t` (稀疏邻接矩阵), `y` (标签)，以及可选的 `r_pe` (位置编码) 和 `r_fourier` (傅里叶特征)。

*   **高斯边权重 (`add_gaussian_edge_weights`)**:
    计算节点间距离并应用高斯核函数生成边权重。
    $$w_{ij} = \exp\left(-\frac{d_{ij}^2}{2\sigma^2}\right)$$

*   **稀疏邻接矩阵构建**:
    将带有高斯权重的边索引转换为 `torch_sparse.SparseTensor` 格式的邻接矩阵。
    $$ \mathbf{A}_{sparse} = \text{SparseTensor}(\text{indices}, \text{values}, \text{matrix\_shape}) $$

*   **拉普拉斯特征向量位置编码 (LapPE) (`AddLaplacianEigenvectorPE`)**:
    如果启用，计算基于图的归一化拉普拉斯矩阵 $\mathbf{L}_{sym}$ 的特征向量作为节点的位置编码。
    $$ \mathbf{L}_{sym} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2} $$
    其中 $k$ 个最小非零特征值对应的特征向量作为位置编码。

*   **图傅里叶特征 (Graph Fourier Features) (`compute_fourier_features`)**:
    如果启用，基于图傅里叶变换提取节点的频率域特征，并生成注意力优化矩阵。核心步骤包括：
    1.  构建归一化拉普拉斯矩阵 $\mathbf{L}$：$$ \mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2} $$
    2.  特征分解 $\mathbf{L} = \mathbf{U} \mathbf{\Lambda} \mathbf{U}^T$。
    3.  图傅里叶变换：$$ \mathbf{X}_f = \mathbf{U}^T \mathbf{X} $$
    4.  低频/高频信号分离：$$ \mathbf{X}_{low} = \mathbf{U} (\mathbf{X}_f \odot \mathbf{M}_{low}) \quad \mathbf{X}_{high} = \mathbf{U} (\mathbf{X}_f \odot \mathbf{M}_{high}) $$
    5.  频率滤波/注意力优化矩阵 (`frequency_filtering`)：
        $$ \mathbf{F}_{ij} = \frac{E_{low}^{(i)} + E_{high}^{(j)}}{\sum_k (E_{low}^{(k)} + E_{high}^{(k)})} \quad \mathbf{M}_{attn} = (\mathbf{\lambda}_i + \mathbf{\lambda}_j) \odot \mathbf{F}_{ij} $$

### 2.2. `preprocess_dataset` 函数

此函数协调整个数据集的预处理，加载原始 `.pkl` 文件，对每个样本调用 `process_single_protein`，然后将处理后的数据保存为新的 `.pkl` 文件 (使用 `torch.save`)。

## 3. 多模态数据整合

经过预处理，得到多个 `.pkl` 文件（例如 `esmc_Train7596.pkl`, `ProtBERT_Train7596.pkl`, `ProtT5_Train7596.pkl`），这些文件被 `DataLoader` 加载，并在 `DualStreamPPI` 模型中进行整合。
