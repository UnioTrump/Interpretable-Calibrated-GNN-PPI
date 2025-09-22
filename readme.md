# DualStreamPPI - 多模态蛋白质-蛋白质相互作用 (PPI) 预测模型

`DualStreamPPI` 是一个用于蛋白质-蛋白质相互作用 (PPI) 预测的图神经网络模型，旨在通过整合多模态蛋白质信息提高预测准确性。模型现在支持最多三个模态的输入，每个模态在 MLP 分类器之前通过两阶段融合机制整合其内部语义与几何信息，再进行跨模态融合。此模型设计支持通过调整输入数据进行消融实验。

## 1. 模型架构 (`GASPPI/model/dual_stream.py`)

`DualStreamPPI` 是模型的核心，采用了**每个模态独立双流结构**，即每个模态包含一个语义特征处理流和一个几何特征处理流，并在模态内部进行初步融合，再进行最终的跨模态融合。

### 1.1. `IntraModalFusion` 类

`IntraModalFusion` 负责在**每个模态内部**将语义流和几何流的嵌入进行融合。它将语义嵌入 $E_{\text{semantic}}$ 和几何嵌入 $E_{\text{geometric}}$ 拼接后，通过一个线性层、ReLU 激活和 Dropout 层输出融合后的模态嵌入 $E_{\text{intra-fused}}$。

#### 数学公式
$$\mathbf{E}_{\text{intra-concat}} = \text{Concat}(\mathbf{E}_{\text{semantic}}, \mathbf{E}_{\text{geometric}})$$
$$\mathbf{E}_{\text{intra-fused}} = \text{Dropout}(\sigma(\mathbf{E}_{\text{intra-concat}} \mathbf{W}_{\text{intra}} + \mathbf{b}_{\text{intra}}))$$
其中：
*   $\mathbf{E}_{\text{semantic}}$ 是语义 GNNEncoder 输出的嵌入。
*   $\mathbf{E}_{\text{geometric}}$ 是几何 GNNEncoder 输出的嵌入。
*   $\mathbf{W}_{\text{intra}}$ 和 $\mathbf{b}_{\text{intra}}$ 是内部融合层的可学习权重和偏置。
*   $\sigma$ 是 ReLU 激活函数。

### 1.2. `MultiModalConcatFusion` 类

`MultiModalConcatFusion` 负责将所有活跃的**模态内部融合后的嵌入** $E_{\text{modal1-fused}}, E_{\text{modal2-fused}}, \dots, E_{\text{modal}M\text{-fused}}$ 进行最终的拼接融合。它将这些模态嵌入拼接后，通过一个线性层、ReLU 激活和 Dropout 层输出最终的融合嵌入 $E_{\text{fused}}$。

#### 数学公式
$$\mathbf{E}_{\text{concat}} = \text{Concat}(\mathbf{E}_{\text{modal1-fused}}, \mathbf{E}_{\text{modal2-fused}}, \dots, \mathbf{E}_{\text{modal}M\text{-fused}})$$
$$\mathbf{E}_{\text{fused}} = \text{Dropout}(\sigma(\mathbf{E}_{\text{concat}} \mathbf{W}_f + \mathbf{b}_f))$$
其中：
*   $\mathbf{E}_{\text{modal}i\text{-fused}}$ 是第 $i$ 个模态内部融合层输出的嵌入。
*   $\mathbf{W}_f$ 和 $\mathbf{b}_f$ 是最终融合层的可学习权重和偏置。
*   $\sigma$ 是 ReLU 激活函数。

### 1.3. `DualStreamPPI` 类

`DualStreamPPI` 类是模型的整体协调器，集成了多个语义和几何 GNNEncoder，并通过两阶段融合机制处理多模态数据。

*   **`__init__`**: 初始化所有组件，包括：
    *   **Feature Stream (模态1)**: `self.feature_semantic_stream` (处理 `data.seq_x`, `data.seq_adj_t`) 和 `self.feature_geometric_stream` (处理 `data.r_pe`, `data.r_fourier`)，以及它们的 `self.feature_intra_fusion` 融合层。
    *   **Modal2 Stream (可选)**: `self.modal2_semantic_stream`, `self.modal2_geometric_stream` 和 `self.modal2_intra_fusion` (如果提供了 `modal2_in_channels` 和 `modal2_pe_dim`)。
    *   **Modal3 Stream (可选)**: `self.modal3_semantic_stream`, `self.modal3_geometric_stream` 和 `self.modal3_intra_fusion` (如果提供了 `modal3_in_channels` 和 `modal3_pe_dim`)。
    *   **`self.fusion`**: `MultiModalConcatFusion` 实例，用于最终的跨模态融合。
    *   **`self.classifier`**: MLP，用于最终的PPI预测。

*   **`forward(self, data)`**:
    *   接收 `data` 对象。
    *   调用 `self.feat(data)` 提取并融合特征。
    *   调用 `self.MLP(fused_embeds)` 得到预测结果。

*   **`feat(self, data)`**:
    *   **模态内部处理与融合**：对于每个活跃模态，分别通过其语义 GNNEncoder 和几何 GNNEncoder 生成嵌入，然后通过对应的 `IntraModalFusion` 层进行融合，得到该模态的内部融合嵌入。
    *   **跨模态融合**：将所有模态内部融合后的嵌入送入 `MultiModalConcatFusion` 进行最终拼接融合，生成 `fused_embeds`。

## 2. 图神经网络编码器 (`GASPPI/model/base.py`)

`GNNEncoder` 是所有模态的语义流和几何流编码器的基础组件。

### 数学公式

#### 通用图卷积层 (GeneralConv / AntiSymmetricConv)

对于节点 $v$，其在 $k+1$ 层的特征 $\mathbf{h}_v^{(k+1)}$ 的更新通常涉及对其邻居 $u \in \mathcal{N}(v)$ 的特征 $\mathbf{h}_u^{(k)}$ 和边信息 $\mathbf{e}_{vu}$ 的聚合，然后通过一个更新函数进行变换：

$$\mathbf{h}_v^{(k+1)} = \text{UPDATE}^{(k)}\left(\mathbf{h}_v^{(k)}, \text{AGGREGATE}^{(k)}\left(\{\mathbf{h}_u^{(k)}, \mathbf{e}_{vu} \mid u \in \mathcal{N}(v)\}\right)\right)$$

其中：
*   $\mathbf{h}_v^{(k)}$ 是节点 $v$ 在第 $k$ 层的特征。
*   $\mathcal{N}(v)$ 是节点 $v$ 的邻居集合。
*   $\text{AGGREGATE}^{(k)}$ 是聚合函数 (例如求和、求平均、最大值等，可能包含边信息)。
*   $\text{UPDATE}^{(k)}$ 是更新函数 (例如 MLP、GRU 等)。

#### 跳跃知识网络 (JumpingKnowledge - JK) - 拼接模式 (Cat)

`JumpingKnowledge` 的拼接模式将所有中间层的节点嵌入进行拼接，以捕获不同尺度的信息。对于一个具有 $L$ 层的 GNNEncoder，其最终输出嵌入 $\mathbf{H}$ 为：

$$\mathbf{H} = \text{Concat}(\mathbf{h}^{(0)}, \mathbf{h}^{(1)}, \dots, \mathbf{h}^{(L)})$$

其中 $\mathbf{h}^{(k)}$ 是第 $k$ 层的节点嵌入。

## 3. 数据加载与预处理 (`data_utils/data_utils.py`)

`DataLoader` 负责加载、组织和准备多模态数据。

*   **`DataLoader.__init__(self, device, multimodal_data_dir)`**:
    *   接收计算设备和多模态 `.pkl` 文件目录路径。
    *   列出目录中所有 `.pkl` 文件，按文件名字母顺序排序（假定顺序与模态 `modal1` (`esmc`), `modal2` (`ProtBERT`), `modal3` (`ProtT5`) 对应）。
    *   使用 `torch.load` 依次加载这三个 `.pkl` 文件到 `self.modalX_list` 中。

*   **`DataLoader.load_data(data_loader_instance)`**:
    *   返回一个整数索引列表 (`range(N)`)，`N` 是样本总数。

*   **`DataLoader.prepare_sample(self, idx)`**:
    *   接收索引 `idx`。
    *   从 `self.modalX_list[idx]` 获取各模态的**语义特征 (`r_node`)、邻接矩阵 (`residue_adj_t`)、标签 (`y`)，以及独立的几何特征 (`r_pe`, `r_fourier`)**。
    *   使用 `move_to_device` 辅助函数将 `torch.Tensor` 和 `torch_sparse.SparseTensor` 移动到指定设备。
    *   封装所有模态数据到 `torch_geometric.data.Data` 对象中，为每个模态的几何特征使用独立的属性名 (例如 `data.modal2_r_pe`, `data.modal3_r_fourier`)。

*   **`DataLoader.get_dat-info(sample_data)`**:
    *   接收 `Data` 对象。
    *   动态返回包含 `sequence_in_channels`, `modal2_in_channels`, `modal3_in_channels` 以及**新增的 `modal2_pe_dim` 和 `modal3_pe_dim`** 的字典，用于模型初始化。

## 4. 模型训练与评估 (`demo.py`)

`demo.py` 是训练和评估流程的入口点。

### 评估指标

在二分类任务中，常用的评估指标基于混淆矩阵中的四个基本量：

*   **真阳性 (True Positives, TP)**：实际为正，预测也为正。
*   **真阴性 (True Negatives, TN)**：实际为负，预测也为负。
*   **假阳性 (False Positives, FP)**：实际为负，预测为正 (I 类错误)。
*   **假阴性 (False Negatives, FN)**：实际为正，预测为负 (II 类错误)。

基于这些基本量，各指标公式如下：

#### 准确率 (Accuracy)

准确率衡量模型正确预测的样本比例。
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

#### 精确率 (Precision)

精确率衡量所有被预测为正的样本中，实际为正的比例。它关注预测结果的“纯度”。
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

#### 召回率 (Recall / Sensitivity)

召回率衡量所有实际为正的样本中，被模型正确预测为正的比例。它关注模型识别正样本的能力。
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

#### 特异性 (Specificity)

特异性衡量所有实际为负的样本中，被模型正确预测为负的比例。它关注模型识别负样本的能力。
$$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$

#### F1 Score

F1 Score 是精确率和召回率的调和平均值，综合考虑了两者的表现。
$$\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

#### 马修斯相关系数 (Matthews Correlation Coefficient, MCC)

MCC 综合考虑了TP, TN, FP, FN，被认为是衡量二分类模型性能的一个更平衡的指标，尤其是在类别不平衡时。
$$\text{MCC} = \frac{\text{TP} \times \text{TN} - \text{FP} \times \text{FN}}{\sqrt{(\text{TP} + \text{FP})(\text{TP} + \text{FN})(\text{TN} + \text{FP})(\text{TN} + \text{FN})}}$$

#### PR-AUC (Precision-Recall Area Under the Curve)

PR-AUC 是精确率-召回率曲线下面积。PR 曲线以召回率为横轴，精确率为纵轴。对于类别不平衡的数据集，PR-AUC 通常比 ROC-AUC 更能反映模型的性能。

#### ROC-AUC (Receiver Operating Characteristic Area Under the Curve)

ROC-AUC 是受试者工作特征曲线下面积。ROC 曲线以假阳性率 (FPR) 为横轴，真阳性率 (TPR) 为纵轴。FPR = FP / (FP + TN) = 1 - Specificity，TPR = Recall。

*   **`main()` 函数**:
    *   设置随机种子。
    *   实例化 `DataLoader` (传入 `config.MULTIMODAL_DATA_DIR`)。
    *   加载数据索引，并拆分为训练集和验证集索引。
    *   获取模型输入维度 (现在包括 `modalX_pe_dim`)，实例化 `DualStreamPPI` 模型。
    *   设置优化器和学习率调度器。
    *   执行训练循环，在每个 epoch 中调用 `train()` 和 `test()` 函数。

*   **`train()` / `test()` 函数**:
    *   处理批次数据，调用 `data_loader.prepare_sample()` 获取 `Data` 对象。
    *   执行模型前向传播，计算损失，并进行反向传播和优化器步进（仅 `train()`）。
    *   在 `test()` 中，收集预测结果并计算评估指标。