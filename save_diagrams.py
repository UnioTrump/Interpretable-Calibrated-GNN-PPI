import os
import requests
import base64
import json

# Create diagrams directory if it doesn't exist
os.makedirs("diagrams", exist_ok=True)

# Mermaid diagram codes
diagram1 = """
flowchart TD
    %% Style definitions with black text
    classDef inputClass fill:#d4ebf2,stroke:#0066cc,stroke-width:2px,color:black;
    classDef atomClass fill:#ffd6cc,stroke:#cc3300,stroke-width:2px,color:black;
    classDef residueClass fill:#d6f5d6,stroke:#339933,stroke-width:2px,color:black;
    classDef mappingClass fill:#fff2cc,stroke:#996600,stroke-width:2px,color:black;
    classDef outputClass fill:#e6ccff,stroke:#6600cc,stroke-width:2px,color:black;
    classDef gatClass fill:#ffe6cc,stroke:#ff9933,stroke-width:1px,color:black;
    classDef historyClass fill:#f2e6ff,stroke:#9966cc,stroke-width:2px,stroke-dasharray: 5 5,color:black;
    classDef poolClass fill:#ffcccc,stroke:#cc6666,stroke-width:1px,stroke-dasharray: 5 5,color:black;
    
    %% Input nodes
    input_atom["Atom Features<br/>(atom_x)"]:::inputClass
    input_atom_adj["Atom Adjacency<br/>(atom_adj_t)"]:::inputClass
    input_res["Residue Features<br/>(residue_x)"]:::inputClass
    input_res_adj["Residue Adjacency<br/>(residue_adj_t)"]:::inputClass
    input_a2r["Atom-to-Residue Mapping<br/>(a2r_map)"]:::inputClass
    
    %% Main diagram
    subgraph "GASPPI: Hierarchical GNN with History Module"
        direction TB
        
        %% Atom Block 1 Details
        subgraph "Atom Block 1 (PPI)"
            direction TB
            a1_gat1["GAT Conv Layer 1<br/>(Multi-head Attention)"]:::gatClass
            a1_elu1["ELU Activation"]:::gatClass
            a1_drop1["Dropout"]:::gatClass
            
            subgraph "History Module 1"
                direction TB
                a1_history["Node Embeddings History<br/>(Historical Embeddings Storage)"]:::historyClass
                a1_push["push: Store Embeddings"]:::historyClass
                a1_pull["pull: Retrieve Embeddings"]:::historyClass
                
                a1_history --> a1_push
                a1_history --> a1_pull
            end
            
            a1_pool["AsyncIOPool<br/>(Asynchronous Memory Transfer)"]:::poolClass
            
            a1_gat2["GAT Conv Layer 2<br/>(Multi-head Attention)"]:::gatClass
            
            a1_gat1 --> a1_elu1 --> a1_drop1
            a1_drop1 -- "Current batch embeddings" --> a1_push
            a1_pull -- "Historical embeddings" --> a1_gat2
            a1_push <--> a1_pool
            a1_pull <--> a1_pool
        end
        
        %% Mapping 1
        a2r_map1["atom2residue<br/>Mapping"]:::mappingClass
        
        %% Concatenation 1
        concat1["Feature<br/>Concatenation"]:::mappingClass
        
        %% Residue Block 1 Details
        subgraph "Residue Block 1 (PPI)"
            direction TB
            r1_gat1["GAT Conv Layer 1<br/>(Multi-head Attention)"]:::gatClass
            r1_elu1["ELU Activation"]:::gatClass
            r1_drop1["Dropout"]:::gatClass
            
            subgraph "History Module 2"
                direction TB
                r1_history["Node Embeddings History<br/>(Historical Embeddings Storage)"]:::historyClass
                r1_push["push: Store Embeddings"]:::historyClass
                r1_pull["pull: Retrieve Embeddings"]:::historyClass
                
                r1_history --> r1_push
                r1_history --> r1_pull
            end
            
            r1_pool["AsyncIOPool<br/>(Asynchronous Memory Transfer)"]:::poolClass
            
            r1_gat2["GAT Conv Layer 2<br/>(Multi-head Attention)"]:::gatClass
            
            r1_gat1 --> r1_elu1 --> r1_drop1
            r1_drop1 -- "Current batch embeddings" --> r1_push
            r1_pull -- "Historical embeddings" --> r1_gat2
            r1_push <--> r1_pool
            r1_pull <--> r1_pool
        end
        
        %% Block separator
        block_sep1["Block 1 Output"]:::residueClass
        
        %% Atom Block 2 Details (Simplified for clarity)
        subgraph "Atom Block 2 (PPI)"
            direction TB
            a2_gat["GAT Layers with<br/>History Module"]:::atomClass
        end
        
        %% Mapping 2
        a2r_map2["atom2residue<br/>Mapping"]:::mappingClass
        
        %% Addition 2
        add2["Feature<br/>Addition"]:::mappingClass
        
        %% Residue Block 2 Details (Simplified for clarity)
        subgraph "Residue Block 2 (PPI)"
            direction TB
            r2_gat["GAT Layers with<br/>History Module"]:::residueClass
        end
        
        %% Block separator
        block_sep2["Block 2 Output"]:::residueClass
        
        %% Block N indication
        blockN_indication["... Additional Blocks ..."]
        
        %% Concatenation of all outputs
        concat_all["Concatenate All Block Outputs<br/>(Skip Connections)"]:::outputClass
        
        %% Output MLP
        subgraph "Output MLP"
            direction TB
            fc1["FC Layer<br/>(out_channels×num_blocks → 128)"]:::outputClass
            relu["ReLU"]:::outputClass
            drop1["Dropout(0.2)"]:::outputClass
            fc2["FC Layer<br/>(128 → 1)"]:::outputClass
            sigmoid["Sigmoid"]:::outputClass
            drop2["Dropout(0.2)"]:::outputClass
            
            fc1 --> relu --> drop1 --> fc2 --> sigmoid --> drop2
        end
        
        output["PPI Prediction"]:::outputClass
        
        %% Scalability Components
        subgraph "Scalability Mechanism"
            direction TB
            scalability["ScalableGNN<br/>(Base Class)"]:::historyClass
            history_mgmt["History Management<br/>(Historical Embeddings)"]:::historyClass
            async_io["AsyncIOPool<br/>(Asynchronous I/O)"]:::poolClass
            
            scalability --> history_mgmt
            scalability --> async_io
            
            note["Note: Each GNN layer uses a History module<br/>to store and retrieve historical embeddings,<br/>enabling efficient training on large graphs"]:::historyClass
        end
    end
    
    %% Connections
    input_atom --> a1_gat1
    input_atom_adj --> a1_gat1
    a1_gat2 --> a2r_map1
    input_res --> concat1
    a2r_map1 --> concat1
    concat1 --> r1_gat1
    input_res_adj --> r1_gat1
    input_a2r --> a2r_map1
    input_a2r --> a2r_map2
    
    r1_gat2 --> block_sep1
    block_sep1 --> add2
    block_sep1 --> concat_all
    
    a1_gat2 --> a2_gat
    a2_gat --> a2r_map2
    a2r_map2 --> add2
    add2 --> r2_gat
    input_res_adj --> r2_gat
    
    r2_gat --> block_sep2
    block_sep2 --> blockN_indication
    block_sep2 --> concat_all
    
    blockN_indication --> concat_all
    concat_all --> fc1
    drop2 --> output
"""

diagram2 = """
flowchart LR
    %% Style definitions with black text
    classDef inputClass fill:#d4ebf2,stroke:#0066cc,stroke-width:2px,color:black;
    classDef atomClass fill:#ffd6cc,stroke:#cc3300,stroke-width:2px,color:black;
    classDef gatClass fill:#ffe6cc,stroke:#ff9933,stroke-width:1px,color:black;
    classDef historyClass fill:#f2e6ff,stroke:#9966cc,stroke-width:2px,color:black;
    classDef poolClass fill:#ffcccc,stroke:#cc6666,stroke-width:1px,color:black;
    classDef memClass fill:#d6f5d6,stroke:#339933,stroke-width:2px,color:black;
    
    %% Main components of the History mechanism
    subgraph "GASPPI History Module Detailed Architecture"
        direction TB
        
        %% GNN Processing
        subgraph "GNN Layer Processing"
            direction TB
            batch_nodes["Current Batch Nodes<br/>(GPU)"]:::inputClass
            gnn_process["GAT Conv Processing<br/>(Multi-head Attention)"]:::gatClass
            batch_output["Processed Batch Embeddings<br/>(GPU)"]:::atomClass
            
            batch_nodes --> gnn_process --> batch_output
        end
        
        %% History Module
        subgraph "History Module"
            direction TB
            
            subgraph "Embedding Storage"
                direction TB
                emb_matrix["Full Graph Embeddings Matrix<br/>(CPU Pinned Memory)"]:::historyClass
            end
            
            %% Push operation
            push_op["push() Operation<br/>Store Current Embeddings"]:::historyClass
            
            %% Pull operation
            pull_op["pull() Operation<br/>Retrieve Historical Embeddings"]:::historyClass
            
            emb_matrix <-- "Update" --> push_op
            emb_matrix <-- "Retrieve" --> pull_op
        end
        
        %% AsyncIO Pool
        subgraph "AsyncIOPool"
            direction TB
            
            %% Asynchronous Push
            async_push["async_push()<br/>Non-blocking Write"]:::poolClass
            sync_push["synchronize_push()<br/>Sync Write Operations"]:::poolClass
            
            %% Asynchronous Pull
            async_pull["async_pull()<br/>Non-blocking Read"]:::poolClass
            sync_pull["synchronize_pull()<br/>Sync Read Operations"]:::poolClass
            
            %% Buffers
            cuda_buf["CUDA Buffers<br/>(GPU)"]:::poolClass
            cpu_buf["CPU Pinned Buffers"]:::poolClass
            
            %% Streams
            push_stream["CUDA Push Streams"]:::poolClass
            pull_stream["CUDA Pull Streams"]:::poolClass
            
            async_push <--> push_stream
            async_pull <--> pull_stream
            push_stream <--> cuda_buf
            pull_stream <--> cuda_buf
            cuda_buf <--> cpu_buf
            
            async_push --> sync_push
            async_pull --> sync_pull
        end
        
        %% Memory layout
        subgraph "Memory Management"
            direction TB
            gpu_mem["GPU Memory<br/>(Limited Capacity, Fast Access)"]:::memClass
            cpu_mem["CPU Memory<br/>(Large Capacity, Slower Access)"]:::memClass
            
            gpu_mem <-- "Asynchronous Transfer" --> cpu_mem
        end
        
        %% Forward pass flow
        forward_flow["1. Forward Pass Flow:<br/>- Process current batch on GPU<br/>- Push embeddings to history (async)<br/>- Pull next batch embeddings (async)<br/>- Continue processing while transfer happens"]:::historyClass
        
        %% Scalability benefit
        scalability["2. Scalability Benefits:<br/>- Can process graphs larger than GPU memory<br/>- Avoids redundant computation<br/>- Optimizes GPU-CPU transfers<br/>- Enables efficient mini-batch training"]:::historyClass
    end
    
    %% Main data flow connections
    batch_output --> push_op
    pull_op --> batch_nodes
    
    push_op --> async_push
    pull_op --> async_pull
    
    sync_push --> emb_matrix
    emb_matrix --> sync_pull
    
    %% Memory management connections
    batch_nodes -.-> gpu_mem
    batch_output -.-> gpu_mem
    emb_matrix -.-> cpu_mem
    cpu_buf -.-> cpu_mem
    cuda_buf -.-> gpu_mem
"""

diagram3 = """
flowchart TD
    %% Style definitions with black text
    classDef inputClass fill:#d4ebf2,stroke:#0066cc,stroke-width:2px,color:black;
    classDef atomClass fill:#ffd6cc,stroke:#cc3300,stroke-width:2px,color:black;
    classDef residueClass fill:#d6f5d6,stroke:#339933,stroke-width:2px,color:black;
    classDef outputClass fill:#e6ccff,stroke:#6600cc,stroke-width:2px,color:black;
    classDef blockClass fill:#fff2cc,stroke:#996600,stroke-width:2px,color:black;
    
    %% Input nodes
    input["Protein Structure Data"]:::inputClass
    
    %% Graph construction
    subgraph "Graph Construction"
        direction TB
        atom_graph["Atom-level Graph Construction<br/>(Fine-grained representation)"]:::atomClass
        residue_graph["Residue-level Graph Construction<br/>(Coarse-grained representation)"]:::residueClass
        a2r_mapping["Atom-to-Residue Mapping"]:::atomClass
        
        atom_graph --- a2r_mapping --- residue_graph
    end
    
    %% Main model
    subgraph "Hierarchical GNN Architecture"
        direction TB
        
        subgraph "Block 1"
            direction LR
            atom_block1["Atom Block 1<br/>GAT + History"]:::atomClass
            a2r_map1["atom2residue<br/>Mapping"]:::atomClass
            res_block1["Residue Block 1<br/>GAT + History"]:::residueClass
            
            atom_block1 --> a2r_map1 --> res_block1
        end
        
        subgraph "Block 2"
            direction LR
            atom_block2["Atom Block 2<br/>GAT + History"]:::atomClass
            a2r_map2["atom2residue<br/>Mapping"]:::atomClass
            res_block2["Residue Block 2<br/>GAT + History"]:::residueClass
            
            atom_block2 --> a2r_map2 --> res_block2
        end
        
        subgraph "Block 3"
            direction LR
            atom_block3["Atom Block 3<br/>GAT + History"]:::atomClass
            a2r_map3["atom2residue<br/>Mapping"]:::atomClass
            res_block3["Residue Block 3<br/>GAT + History"]:::residueClass
            
            atom_block3 --> a2r_map3 --> res_block3
        end
        
        %% Skip connections
        skip_connections["Skip Connections<br/>(Concatenate all block outputs)"]:::blockClass
        
        %% Output processing
        mlp["MLP Classifier<br/>FC(out_channels*num_blocks, 128)<br/>FC(128, 1)"]:::outputClass
    end
    
    %% Output
    output["PPI Prediction"]:::outputClass
    
    %% Connections
    input --> atom_graph
    input --> residue_graph
    
    atom_graph --> atom_block1
    residue_graph --> res_block1
    a2r_mapping --> a2r_map1
    a2r_mapping --> a2r_map2
    a2r_mapping --> a2r_map3
    
    res_block1 --> atom_block2
    res_block1 --> skip_connections
    
    res_block2 --> atom_block3
    res_block2 --> skip_connections
    
    res_block3 --> skip_connections
    
    skip_connections --> mlp --> output
    
    %% Block connections
    atom_block1 --> atom_block2 --> atom_block3
"""

# Function to save Mermaid diagram as image
def save_mermaid_diagram(diagram_code, output_path, img_format="png"):
    # Encode the Mermaid diagram
    encoded_diagram = base64.b64encode(diagram_code.strip().encode('utf-8')).decode('utf-8')
    
    # Use mermaid.ink service
    url = f"https://mermaid.ink/{img_format}/{encoded_diagram}"
    
    try:
        # Download the image
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"Successfully saved diagram to {output_path}")
            return True
        else:
            print(f"Failed to download image: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading image: {e}")
        return False

try:
    # Save the diagrams
    save_mermaid_diagram(diagram1, "diagrams/hierarchical_gnn.png")
    save_mermaid_diagram(diagram2, "diagrams/history_module.png")
    save_mermaid_diagram(diagram3, "diagrams/model_overview.png")
    
    print("\nAll diagrams saved successfully in the 'diagrams' directory.")
except Exception as e:
    print(f"Error saving diagrams: {e}") 