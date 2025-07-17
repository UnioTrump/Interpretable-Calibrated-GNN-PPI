def create_model_diagram():
    """
    生成简化版 ProteinGNN 主结构图，每个大模块用一个块表示。
    """
    mermaid_definition = r"""
flowchart TD
    A["Atom Features"] --> AtomGNN["Atom-level GNN"]
    R["Residue Features"] --> Cat["Concat"]
    AtomGNN --> Pool["Global Mean Pooling"] --> Cat
    Cat --> ResidueGNN["Residue-level GNN"]
    ResidueGNN --> Classifier["MLP Classifier"]
    Classifier --> Output["Prediction (Residue-level)"]

    %% --- 样式美化 ---
    style A fill:#cde,stroke:#333,stroke-width:2px
    style R fill:#cde,stroke:#333,stroke-width:2px
    style AtomGNN fill:#f9f,stroke:#333,stroke-width:2px
    style Pool fill:#fdc,stroke:#333,stroke-width:2px
    style Cat fill:#eee,stroke:#333,stroke-width:2px
    style ResidueGNN fill:#f9f,stroke:#333,stroke-width:2px
    style Classifier fill:#9cf,stroke:#333,stroke-width:2px
    style Output fill:#9c9,stroke:#333,stroke-width:4px
    """
    return mermaid_definition


def create_gnn_block_diagram():
    """
    生成美化版GNN内部结构细节图（以Atom-level GNN为例），适合PPT细节页展示。
    """
    mermaid_definition = r"""
flowchart LR
    %% --- 输入 ---
    X(["<b>Input Features</b>"]):::input --> TC1["<b>TransformerConv</b><br/>+ <i>Edge Weights</i>"]:::conv
    TC1 -.->|"<b>Edge Weights<br/>(Gaussian)</b>"| TC1
    TC1 --> LN1["<b>LayerNorm</b>"]:::norm --> DO1["<b>Dropout</b>"]:::drop
    DO1 --> TC2["<b>TransformerConv</b>"]:::conv --> LN2["<b>LayerNorm</b>"]:::norm --> DO2["<b>Dropout</b>"]:::drop
    DO2 --> TC3["<b>TransformerConv</b>"]:::conv --> LN3["<b>LayerNorm</b>"]:::norm --> DO3["<b>Dropout</b>"]:::drop
    DO3 --> JK["<b>JumpingKnowledge</b><br/><i>(Concat)</i>"]:::jk --> Out(["<b>Output Embedding</b>"]):::output

    %% --- 分组区块 ---
    subgraph GNNBlock["<b>GNN Block (Atom-level)</b>"]
        direction LR
        X --> TC1 --> LN1 --> DO1 --> TC2 --> LN2 --> DO2 --> TC3 --> LN3 --> DO3 --> JK --> Out
    end

    %% --- 样式定义 ---
    classDef input fill:#cde,stroke:#333,stroke-width:2px,font-weight:bold;
    classDef conv fill:#f9f,stroke:#333,stroke-width:2px;
    classDef norm fill:#eee,stroke:#333,stroke-width:2px;
    classDef drop fill:#fdc,stroke:#333,stroke-width:2px;
    classDef jk fill:#9cf,stroke:#333,stroke-width:2px,font-weight:bold;
    classDef output fill:#9c9,stroke:#333,stroke-width:3px,font-weight:bold;
    classDef edgeweight fill:#fff,stroke:#f66,stroke-width:2px;

    %% --- 图标注释 ---
    %% 这里可根据PPT需要添加说明
    %% 注：TransformerConv支持边权重，JumpingKnowledge融合多层特征
    """
    return mermaid_definition

if __name__ == "__main__":
    diagram = create_model_diagram()
    print("========= Model Architecture Diagram (Mermaid Syntax) =========")
    print("\nCopy the text below and paste it into a Mermaid Live Editor or compatible tool.\n")
    print(diagram)
    print("\n=================================================================")
    # 比如: https://mermaid.live/
