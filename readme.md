# DualStreamPPI - 多模态蛋白质-蛋白质相互作用 (PPI) 预测模型

`DualStreamPPI` 是一个用于多模态蛋白质-蛋白质相互作用 (PPI) 预测的图神经网络模型。它支持最多三个模态输入，通过双流结构（语义与几何）和两阶段融合机制（模态内融合与跨模态融合）提高预测准确性。

## 1. 模型架构 (`GASPPI/model/dual_stream.py`)

`DualStreamPPI` 是模型的核心，采用了**每个模态独立双流结构**，即每个模态包含一个语义特征处理流和一个几何特征处理流，并在模态内部进行初步融合，再进行最终的跨模态融合。

### 1.1. `IntraModalFusion` 类

`IntraModalFusion` 负责在每个模态内部融合语义流 $E_{\text{semantic}}$ 和几何流 $E_{\text{geometric}}$ 的嵌入，通过线性层、ReLU 和 Dropout 生成融合后的模态嵌入 $E_{\text{intra-fused}}$。

#### 数学公式
$$\mathbf{E}_{\text{intra-concat}} = \text{Concat}(\mathbf{E}_{\text{semantic}}, \mathbf{E}_{\text{geometric}})$$
$$\mathbf{E}_{\text{intra-fused}} = \text{Dropout}(\sigma(\mathbf{E}_{\text{intra-concat}} \mathbf{W}_{\text{intra}} + \mathbf{b}_{\text{intra}}))$$
其中：
*   $\mathbf{E}_{\text{semantic}}$ 是语义 GNNEncoder 输出的嵌入。
*   $\mathbf{E}_{\text{geometric}}$ 是几何 GNNEncoder 输出的嵌入。
*   $\mathbf{W}_{\text{intra}}$ 和 $\mathbf{b}_{\text{intra}}$ 是内部融合层的可学习权重和偏置。
*   $\sigma$ 是 ReLU 激活函数。

### 1.2. `MultiModalConcatFusion` 类

`MultiModalConcatFusion` 负责将所有模态内部融合后的嵌入进行拼接，并通过线性层、ReLU 和 Dropout 输出最终的融合嵌入 $E_{\text{fused}}$。

#### 数学公式
$$\mathbf{E}_{\text{concat}} = \text{Concat}(\mathbf{E}_{\text{modal1-fused}}, \mathbf{E}_{\text{modal2-fused}}, \dots, \mathbf{E}_{\text{modal}M\text{-fused}})$$
$$\mathbf{E}_{\text{fused}} = \text{Dropout}(\sigma(\mathbf{E}_{\text{concat}} \mathbf{W}_f + \mathbf{b}_f))$$
其中：
*   $\mathbf{E}_{\text{modal}i\text{-fused}}$ 是第 $i$ 个模态内部融合层输出的嵌入。
*   $\mathbf{W}_f$ 和 $\mathbf{b}_f$ 是最终融合层的可学习权重和偏置。
*   $\sigma$ 是 ReLU 激活函数。

### 1.3. `DualStreamPPI` 类

`DualStreamPPI` 类是模型的整体协调器，集成了多个语义和几何 GNNEncoder，并通过两阶段融合机制处理多模态数据。

*   **`__init__`**: 初始化各模态的双流编码器、模态内融合层、跨模态融合层和最终分类器 (MLP)。
*   **`forward(self, data)`**: 调用 `self.feat(data)` 提取融合特征，再通过 `self.MLP` 进行预测。
*   **`feat(self, data)`**: 实现模态内部处理与融合，以及最终的跨模态融合。

## 2. 图神经网络编码器 (`GASPPI/model/base.py`)

`GNNEncoder` 是所有模态的语义流和几何流编码器的基础组件。

### 数学公式

#### AntiSymmetricConv

根据 [https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.AntiSymmetricConv.html](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.AntiSymmetricConv.html) 的描述，`AntiSymmetricConv` 是基于“Anti-Symmetric DGN: a stable architecture for Deep Graph Networks”论文的图卷积算子。其核心数学公式为：

$$\mathbf{x}^{\prime}_i = \mathbf{x}_i + \epsilon \cdot \sigma \left( (\mathbf{W}-\mathbf{W}^T-\gamma \mathbf{I}) \mathbf{x}_i + \Phi(\mathbf{X}, \mathcal{N}_i) + \mathbf{b}\right)$$
其中：
*   $\mathbf{x}^{\prime}_i$ 是节点 $i$ 更新后的特征。
*   $\mathbf{x}_i$ 是节点 $i$ 原始特征。
*   $\epsilon$ 是离散化步长 (discretization step size)。
*   $\sigma$ 是非线性激活函数 (e.g., tanh)。
*   $\mathbf{W}$ 是可学习的权重矩阵。
*   $\mathbf{W}^T$ 是 $ \mathbf{W} $ 的转置。
*   $\gamma$ 是扩散强度 (strength of the diffusion)，调节方法的稳定性。
*   $\mathbf{I}$ 是单位矩阵。
*   $\Phi(\mathbf{X}, \mathcal{N}_i)$ 表示一个 `MessagePassing` 层，它聚合了节点 $i$ 邻居 $ \mathcal{N}_i $ 的信息。
*   $\mathbf{b}$ 是偏置项 (bias)。

#### GeneralConv

根据 [https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GeneralConv.html](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GeneralConv.html) 的描述，`GeneralConv` 是一个通用的 GNN 层，改编自“Design Space for Graph Neural Networks”论文。它的灵活性体现在其丰富的参数配置上，包括聚合方案 (`aggr`)、是否支持有向消息传递 (`directed_msg`)、多头机制 (`heads`)、注意力机制 (`attention`) 等。

其基础可以概括为以下通用的图卷积公式：

对于节点 $v$，其新的特征 $\mathbf{h}_v^{(l+1)}$ 可以通过对自身特征的转换以及从邻居 $u \in \mathcal{N}(v)$ 接收到的消息进行聚合来计算：

$$\mathbf{h}_v^{(l+1)} = \text{UPDATE} \left( \mathbf{h}_v^{(l)}, \text{AGGREGATE}_{u \in \mathcal{N}(v)} \left( \text{MESSAGE}(\mathbf{h}_v^{(l)}, \mathbf{h}_u^{(l)}, \mathbf{e}_{vu}) \right) \right)$$
其中：
*   $ \text{MESSAGE} $ 函数结合了当前节点、邻居节点和边特征来生成消息。
*   $ \text{AGGREGATE} $ 函数汇聚了所有传入的消息 (由 `aggr` 参数控制)。
*   $ \text{UPDATE} $ 函数结合了聚合后的消息和节点自身特征来更新节点特征。

#### 跳跃知识网络 (JumpingKnowledge - JK) - 拼接模式 (Cat)

`JumpingKnowledge` 的拼接模式将所有中间层的节点嵌入进行拼接，以捕获不同尺度的信息。对于一个具有 $L$ 层的 GNNEncoder，其最终输出嵌入 $\mathbf{H}$ 为：

$$\mathbf{H} = \text{Concat}(\mathbf{h}^{(0)}, \mathbf{h}^{(1)}, \dots, \mathbf{h}^{(L)})$$
其中 $\mathbf{h}^{(k)}$ 是第 $k$ 层的节点嵌入。

## 3. 数据加载与预处理 (`data_utils/data_utils.py`)

`DataLoader` 负责加载、组织和准备多模态数据。

*   **`DataLoader.__init__`**: 初始化数据加载器，接收设备和多模态数据目录，并加载 `.pkl` 文件。
*   **`DataLoader.load_data`**: 返回数据索引列表。
*   **`DataLoader.prepare_sample`**: 根据索引准备 `torch_geometric.data.Data` 样本，包括语义特征、邻接矩阵、标签和几何特征。
*   **`DataLoader.get_dat-info`**: 返回模型初始化所需的数据信息字典。

## 4. 模型训练与评估 (`demo.py`)

`demo.py` 是训练和评估流程的入口点。

### 评估指标

评估指标基于混淆矩阵中的 TP, TN, FP, FN。
*   **真阳性 (True Positives, TP)**：实际和预测均为正。
*   **真阴性 (True Negatives, TN)**：实际和预测均为负。
*   **假阳性 (False Positives, FP)**：实际为负，预测为正。
*   **假阴性 (False Negatives, FN)**：实际为正，预测为负。

基于这些基本量，各指标公式如下：

#### 准确率 (Accuracy)

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
#### 精确率 (Precision)

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
#### 召回率 (Recall / Sensitivity)

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
#### 特异性 (Specificity)

$$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$
#### F1 Score

$$\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
#### 马修斯相关系数 (Matthews Correlation Coefficient, MCC)

$$\text{MCC} = \frac{\text{TP} \times \text{TN} - \text{FP} \times \text{FN}}{\sqrt{(\text{TP} + \text{FP})(\text{TP} + \text{FN})(\text{TN} + \text{FP})(\text{TN} + \text{FN})}}$$
#### PR-AUC (Precision-Recall Area Under the Curve)

PR-AUC 是精确率-召回率曲线下面积，适用于类别不平衡数据集。

#### ROC-AUC (Receiver Operating Characteristic Area Under the Curve)

ROC-AUC 是受试者工作特征曲线下面积。

*   **`main()` 函数**: 初始化数据、模型、优化器和调度器，并运行训练循环，在每个 epoch 调用 `train()` 和 `test()`。
*   **`train()` / `test()` 函数**: `train()` 处理批量数据、模型前向传播、损失计算和反向传播；`test()` 进行模型评估并计算指标。