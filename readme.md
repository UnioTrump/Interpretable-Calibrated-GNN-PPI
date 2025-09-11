`DualStreamPPI` 双流图神经网络模型，专门用于蛋白质-蛋白质相互作用 (PPI) 预测。该模型通过两个并行流分别处理蛋白质的**特征信息**和**几何拓扑信息**，并通过门控融合机制进行信息整合，最终输出PPI预测结果。

#### DualStreamPPI (`GASPPI/model/dual_stream.py`)

`DualStreamPPI` 是模型的核心，其设计理念是通过两个独立的编码器处理不同类型的蛋白质表示，然后将它们融合。

*   **`__init__`**:
    *   **`feature_stream`**: 一个 `ProteinGNN` 实例，负责处理蛋白质的原子和残基特征。
    *   **`geometric_stream`**: 一个 `GNNEncoder` 实例，负责处理蛋白质的几何（位置编码）信息。
    *   **`fusion`**: 一个 `GatedFusion` 实例，用于智能地融合两个流的输出。
    *   **`classifier`**: 一个简单的全连接层序列，用于将融合后的特征映射到最终的PPI预测分数。
*   **`forward(self, data)`**:
    *   接收一个 `data` 对象（`torch_geometric.data.Data` 实例），其中包含原子特征 (`atom_x`), 原子邻接矩阵 (`atom_adj_t`), 残基特征 (`residue_x`), 残基邻接矩阵 (`residue_adj_t`), 原子到残基映射 (`a2r_map`), 几何位置编码 (`r_pe`) 和傅里叶表示 (`r_fourier`)。
    *   `feature_stream` 处理原子和残基特征。
    *   `geometric_stream` 处理几何位置编码和傅里叶表示（作为几何邻接信息）。
    *   `GatedFusion` 将两个流的输出进行加权融合，通过一个门控机制动态调整每个流的重要性。
    *   最终，融合后的特征通过 `classifier` 得到预测结果。

*   **`GatedFusion` 类**:
    *   实现了门控融合机制。它学习如何结合两个输入流（`feature_embeds` 和 `geo_embeds`），通过一个 sigmoid 激活的门控单元来加权两个流的贡献。这使得模型可以根据输入的具体情况，自适应地调整对特征和几何信息的侧重。

#### 图神经网络编码器：GNNEncoder 与 ProteinGNN (`GASPPI/model/base.py`)

模型的基石是图神经网络编码器，它们定义了如何从图结构数据中提取特征。

*   **`GNNEncoder` 类**:
    *   一个通用的 GNN 编码器，基于 `TransformerConv` 层构建。
    *   **`__init__`**:
        *   使用 `Linear` 层进行初始输入投影。
        *   包含一个 `ModuleList` 存储多个 `TransformerConv` 层，这些层可以捕获节点之间复杂的依赖关系。
        *   `LayerNorm` 和 `Dropout` 用于稳定训练和防止过拟合。
        *   `JumpingKnowledge (JK)` 网络用于聚合所有中间层的输出，这有助于捕获不同尺度下的特征信息并提升模型性能。
    *   **`forward(self, x: Tensor, adj_t: SparseTensor)`**:
        *   处理输入节点特征 `x` 和邻接信息 `adj_t`。
        *   通过一系列 `TransformerConv` 层进行消息传递和特征更新。
        *   在每个卷积层后应用 ReLU 激活、层归一化和 dropout。
        *   最终，使用 `JumpingKnowledge` 将所有中间层的输出拼接起来作为最终的节点嵌入。该编码器支持处理 `SparseTensor` 稀疏邻接矩阵和稠密邻接矩阵。

*   **`ProteinGNN` 类**:
    *   基于 `GNNEncoder` 构建，专门用于处理蛋白质的原子和残基层级的图结构。
    *   **`__init__`**:
        *   包含一个 `atom_encoder` (GNNEncoder 实例)，处理原子图。
        *   包含一个 `residue_encoder` (GNNEncoder 实例)，处理残基图。
    *   **`forward`**:
        *   首先，`atom_encoder` 处理原子特征和原子邻接矩阵，得到原子级别的嵌入。
        *   然后，使用 `global_mean_pool` 将原子级别的嵌入聚合到其所属的残基上（通过 `atom_to_residue_map`），生成池化后的原子特征。
        *   这些池化后的原子特征与原始残基特征拼接起来，作为 `residue_encoder` 的输入。
        *   `residue_encoder` 处理结合了原子信息的残基特征和残基邻接矩阵，最终输出残基级别的嵌入，作为蛋白质的整体特征表示。


--------------------
  Roc Auc: 0.7387
  Pr Auc: 0.3659
  Accuracy: 0.6194
  Precision: 0.2911
  Recall: 0.7631
  Specificity: 0.5875
  F1 Score: 0.4215
  Mcc: 0.2707
  Threshold: 0.1717
  Confusion Matrix:
    TN: 5104
    FP: 3584
    FN: 457
    TP: 1472
--------------------