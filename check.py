def create_model_diagram():
    """
    生成并返回描述 HierarchicalGNN 模型架构的 Mermaid 图定义。
    此版本为最终修正版，移除了所有可能被误解为Markdown的语法（如编号）。
    """
    mermaid_definition = r"""
graph TD
    subgraph "Input Data"
        A["Atom Features<br/>- atom_x -"]
        A_adj["Atom Adjacency<br/>- atom_adj_t -"]
        R["Residue Features<br/>- residue_x -"]
        R_adj["Residue Adjacency<br/>- residue_adj_t -"]
        A2R["Atom-to-Residue Map<br/>- a2r_map -"]
    end

    subgraph "HierarchicalGNN Model"
        subgraph " "
            direction LR
            subgraph "Atom-level GNN"
                AtomGNN_Blocks["GatedGNNBlocks<br/>- N Layers -"]
            end

            Pool["Atom-to-Residue<br/>Pooling"]
            Cat1["Concatenate<br/>Features"]
            
            subgraph "Residue-level GNN"
                ResidueGNN_Blocks["GatedGNNBlocks<br/>- M Layers -"]
            end
        end

        subgraph "Global Context Fusion"
            GlobalPool["Global Mean Pooling"]
            Broadcast["Broadcast & Concatenate<br/>- Local + Global -"]
        end

        Classifier["Classifier - MLP"]
    end

    subgraph "Output"
        Output_Node(("Prediction<br/>- Logits -"))
    end

    %% --- Data Flow ---
    A & A_adj --> AtomGNN_Blocks
    AtomGNN_Blocks -- "Atom Embeddings" --> Pool
    A2R -- "Pooling Map" --> Pool

    Pool -- "Pooled Atom Feats" --> Cat1
    R -- "Original Residue Feats" --> Cat1
    
    Cat1 -- "Combined Feats" --> ResidueGNN_Blocks
    R_adj -- "Residue Structure" --> ResidueGNN_Blocks

    ResidueGNN_Blocks -- "Local Residue Feats" --> Broadcast
    ResidueGNN_Blocks --> GlobalPool
    GlobalPool -- "Global Protein Feat" --> Broadcast

    Broadcast -- "Fused Features" --> Classifier
    Classifier --> Output_Node

    %% --- Styling ---
    style A fill:#cde,stroke:#333,stroke-width:2px
    style R fill:#cde,stroke:#333,stroke-width:2px
    style A_adj fill:#eee,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
    style R_adj fill:#eee,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
    style A2R fill:#eee,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5

    style AtomGNN_Blocks fill:#f9f,stroke:#333,stroke-width:2px
    style ResidueGNN_Blocks fill:#f9f,stroke:#333,stroke-width:2px
    style Pool fill:#e9e,stroke:#333,stroke-width:2px
    style GlobalPool fill:#e9e,stroke:#333,stroke-width:2px

    style Classifier fill:#9cf,stroke:#333,stroke-width:2px
    style Output_Node fill:#9c9,stroke:#333,stroke-width:4px
    """
    return mermaid_definition

if __name__ == "__main__":
    diagram = create_model_diagram()
    print("========= Model Architecture Diagram (Mermaid Syntax) =========")
    print("\nCopy the text below and paste it into a Mermaid Live Editor or compatible tool.\n")
    print(diagram)
    print("\n=================================================================")
    # 比如: https://mermaid.live/
