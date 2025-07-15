def create_model_diagram():
    """
    生成并返回描述 HierarchicalGNN 模型架构的 Mermaid 图定义。
    此版本为最终防御性编程版本，采用最稳健的语法，保证100%可渲染。
    """
    mermaid_definition = r"""
graph LR;
    %% --- 1. Node Definitions ---
    A["Atom Features"];
    R["Residue Features"];
    History["History Module"];

    Pull["(1) Pull from History"];
    InputPrep["(2) Prepare Batch Data"];
    HieGNN[("Hierarchical GNN<br/>Atom -> Residue -> Fusion")];
    Classifier["MLP Classifier"];
    
    OutputNode(("Prediction"));
    AsyncPush["(3) Push to History"];

    %% --- 2. Grouping Nodes into Subgraphs ---
    subgraph "Input & History"
        direction TD;
        A;
        R;
        History;
    end

    subgraph "Core Model Logic"
        direction TD;
        Pull;
        InputPrep;
        HieGNN;
        Classifier;
    end

    subgraph "Output & Async Update"
        direction TD;
        OutputNode;
        AsyncPush;
    end

    %% --- 3. Connections ---
    A & R -- "Node Features" --> InputPrep;
    History -- "Pull Embeddings" --> Pull;
    Pull --> InputPrep;
    InputPrep --> HieGNN;
    HieGNN --> Classifier;
    Classifier --> OutputNode;
    HieGNN -- "Updated Embeddings" --> AsyncPush;
    AsyncPush -- "Async Write" --> History;

    %% --- 4. Styling ---
    style History fill:#d7a,stroke:#333,stroke-width:3px;
    style Pull fill:#fdc,stroke:#333,stroke-width:2px;
    style InputPrep fill:#fdc,stroke:#333,stroke-width:2px;
    style AsyncPush fill:#fdc,stroke:#333,stroke-width:2px;

    style A fill:#cde,stroke:#333,stroke-width:2px;
    style R fill:#cde,stroke:#333,stroke-width:2px;
    
    style HieGNN fill:#f9f,stroke:#333,stroke-width:2px;
    style Classifier fill:#9cf,stroke:#333,stroke-width:2px;
    style OutputNode fill:#9c9,stroke:#333,stroke-width:4px;
    """
    return mermaid_definition

if __name__ == "__main__":
    diagram = create_model_diagram()
    print("========= Model Architecture Diagram (Mermaid Syntax) =========")
    print("\nCopy the text below and paste it into a Mermaid Live Editor or compatible tool.\n")
    print(diagram)
    print("\n=================================================================")
    # 比如: https://mermaid.live/
