import torch
import os
from model import PPI
import config
from tqdm import tqdm
from Data import PPIData, PPIDataset, sparse_collate
import numpy as np
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx

from torch_geometric.explain import Explainer, GNNExplainer
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

device = config.DEVICE


def _load_model(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model = PPI(hid_dim=config.gcn_hid_dim, heads=config.HEADS, dropout=config.DROPOUT).to(device)
    model.load_state_dict(checkpoint['model'])
    T = checkpoint['T'].to(device)
    return model, T


class ExplainWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, edge_attr):
        return self.model.Explain(x, edge_index, edge_attr)

def build_explainer(model):
    return Explainer(
        model=ExplainWrapper(model),
        algorithm=GNNExplainer(epochs=50),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification',
            task_level='node',
            return_type='raw',
        )
    )

def explain_first_binding_site(model, val_loader, explainer, save_dir='./'):
    """For each protein, explain the prediction of the **first** binding-site residue."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'SubGraph.csv'), 'w', encoding='utf-8') as f:
        f.write('PID\tTargetRes\tTopPct\tSubNodes\tBS_in_Sub\tBS_ratio\tTotal_BS\n')

    for batch in tqdm(val_loader, desc='Explaining'):
        if batch['pid'][0] == '4BH6' or batch['pid'][0] == '6GR8':
            batch = {k: v.to(config.DEVICE, non_blocking=True) if (torch.is_tensor(v) or hasattr(v, 'to')) else v
                     for k, v in batch.items()}

            x, edge_index, edge_attr = model.Fuse(
                ax=batch['AA'], bx=batch['esm_c'], cx=batch['dssp'],
                dx=batch['BLOSUM'], ex=batch['pse'],
                fx=batch['res_atom'], adj=batch['adj']
            )
            x = x.detach().requires_grad_(True)
            edge_attr = edge_attr.detach().requires_grad_(True)

            labels = batch['y'].cpu()
            # 找到该蛋白的第一个 binding-site 残基
            interface_idx = (labels == 1).nonzero(as_tuple=True)[0]
            if len(interface_idx) == 0:
                continue
            first_interface = int(interface_idx[18])

            # GNNExplainer: 学习边掩码 M ∈ [0,1]^E
            # 扰动方式: msg = msg * M, 优化目标: 最大化 I(Y; M⊙G)
            explanation = explainer(
                x=x, edge_index=edge_index, edge_attr=edge_attr,
                index=first_interface,
            )

            edge_mask = explanation.edge_mask.detach().cpu()
            E = edge_mask.size(0)
            edge_index_cpu = edge_index.cpu()
            y_np = labels.numpy()
            pid = batch['pid'][0]
            total_bs_count = int((y_np == 1).sum())

            # --- Node importance: computed from the FULL edge mask (all E edges),
            #     NOT from the subgraph — so it stays consistent across thresholds.
            #     Each node's score = sum of mask values of all its incident edges. ---
            full_node_importance = {}
            for e in range(E):
                u, v = int(edge_index_cpu[0, e]), int(edge_index_cpu[1, e])
                w = float(edge_mask[e])
                full_node_importance[u] = full_node_importance.get(u, 0.0) + w
                full_node_importance[v] = full_node_importance.get(v, 0.0) + w

            max_ni = max(full_node_importance.values()) if full_node_importance else 1.0
            min_ni = min(full_node_importance.values()) if full_node_importance else 0.0
            ni_range = max_ni - min_ni if max_ni > min_ni else 1.0

            # --- Pre-build all three subgraphs (5%, 10%, 20%) ---
            graphs = {}        # top_pct -> (G, node_list, edge_list_for_drawing)
            for top_pct in [0.05, 0.10, 0.20]:
                k = max(1, int(E * top_pct))
                _, topk_idx = torch.topk(edge_mask, k)
                sub_edges = edge_index_cpu[:, topk_idx]
                sub_edge_masks = edge_mask[topk_idx]
                sub_nodes_set = set(sub_edges[0].tolist() + sub_edges[1].tolist())

                G = nx.Graph()
                for i in range(sub_edges.shape[1]):
                    u, v = int(sub_edges[0, i]), int(sub_edges[1, i])
                    G.add_edge(u, v, weight=float(sub_edge_masks[i]))
                if first_interface not in G:
                    G.add_node(first_interface)
                    sub_nodes_set.add(first_interface)
                for n in sub_nodes_set:
                    if n not in G:
                        G.add_node(n)

                # Per-edge drawing data: (u, v, mask_value)
                edge_draw = []
                node_order = list(G.nodes())
                for i in range(sub_edges.shape[1]):
                    edge_draw.append((int(sub_edges[0, i]), int(sub_edges[1, i]),
                                      float(sub_edge_masks[i])))

                graphs[top_pct] = (G, node_order, edge_draw)

            # --- Layout: compute ONCE on the 20 % graph, share with 5 % / 10 % ---
            G20, _, _ = graphs[0.20]
            pos = nx.spring_layout(
                G20, weight='weight', seed=42, k=1.2, iterations=100,
                fixed=[first_interface],
                pos={first_interface: np.array([0.0, 0.0])},
            )

            # --- Node size: sqrt scaling (plan §4) for better visual spread ---
            def _node_score(n):
                return max(0.0, (full_node_importance.get(n, 0.0) - min_ni) / ni_range)

            # --- Draw each threshold using the SAME layout ---
            for top_pct in [0.05, 0.10, 0.20]:
                G, node_list, edge_draw = graphs[top_pct]

                sub_nodes = list(G.nodes())
                sub_bs_count = int(y_np[sub_nodes].sum())
                sub_bs_ratio = sub_bs_count / len(sub_nodes) if len(sub_nodes) > 0 else 0

                # Write CSV row
                with open(os.path.join(save_dir, 'SubGraph.csv'), 'a', encoding='utf-8') as f:
                    f.write(f'{pid}\t{first_interface}\t{int(top_pct*100)}%\t{len(sub_nodes)}\t{sub_bs_count}\t{sub_bs_ratio:.4f}\t{total_bs_count}\n')

                # sqrt-scaled node sizes (plan §4)
                node_sizes = [100.0 + 400 * _node_score(n)**2 for n in node_list]

                # Node fill + border colors
                node_colors, node_edgecolors = [], []
                for n in node_list:
                    if n == first_interface:
                        node_colors.append('#2166ac')
                        node_edgecolors.append('#08306b')
                    elif y_np[n] == 1:
                        node_colors.append('#b2182b')
                        node_edgecolors.append('#67000d')
                    else:
                        node_colors.append('#e0e0e0')
                        node_edgecolors.append('#aaaaaa')

                # --- Draw ---
                fig, ax = plt.subplots(figsize=(8, 8))

                # Edge width + colour both mapped to mask (plan §1)
                # matplotlib doesn't support per-edge alpha, so we encode
                # importance as grayscale: low mask → light, high mask → dark.
                masks = np.array([e[2] for e in edge_draw])
                if len(masks) > 0:
                    m_min, m_max = masks.min(), masks.max()
                    m_range = m_max - m_min if m_max > m_min else 1.0
                else:
                    m_min, m_max, m_range = 0.0, 1.0, 1.0

                edge_widths = [0.5 + 4.0 * (e[2] - m_min) / m_range for e in edge_draw]
                edge_colors = []
                for e in edge_draw:
                    t = (e[2] - m_min) / m_range  # 0 (low mask) → 1 (high mask)
                    r = int(230 + (183 - 230) * t)  # #E6 → #B7
                    g = int(238 + (208 - 238) * t)  # #EE → #D0
                    b = int(245 + (234 - 245) * t)  # #F5 → #EA
                    edge_colors.append(f'#{r:02x}{g:02x}{b:02x}')

                # Draw edges one by one (needed for per-edge width + colour)
                for (u, v, _), w, c in zip(edge_draw, edge_widths, edge_colors):
                    if u in pos and v in pos:
                        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                                linewidth=w, color=c, alpha=0.5, solid_capstyle='round')

                # Nodes
                nx.draw_networkx_nodes(G, pos, nodelist=node_list,
                                       node_size=node_sizes, node_color=node_colors,
                                       edgecolors=node_edgecolors, linewidths=0.8, ax=ax)

                # Labels: target + top-3 nodes in THIS subgraph (global importance rank)
                local_imp = sorted(
                    [(n, full_node_importance.get(n, 0.0)) for n in G.nodes() if n != first_interface],
                    key=lambda x: x[1], reverse=True,
                )
                top3_local = [n for n, _ in local_imp[:3]]
                label_set = {first_interface} | set(top3_local)
                labels = {n: str(n) for n in label_set if n in pos}
                nx.draw_networkx_labels(G, pos, labels=labels, font_size=7,
                                        font_color='#333333', ax=ax)

                # Node-type legend
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2166ac',
                           markersize=10, label=f'Target residue ({first_interface})'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#b2182b',
                           markersize=10, label='Binding-site residues'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e0e0e0',
                           markersize=10, label='Non-binding residues'),
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=7,
                          framealpha=0.9, edgecolor='#cccccc')

                pct_label = f'{int(top_pct * 100)}%'
                ax.set_title(f'{pid} — Subgraph (top-{pct_label} edges) for residue {first_interface}\n'
                             f'(Nodes: {len(sub_nodes)}, BS in subgraph: {sub_bs_count}/{len(sub_nodes)} = {sub_bs_ratio:.1%})',
                             fontsize=9)
                ax.axis('off')
                plt.tight_layout()

                plt.savefig(os.path.join(save_dir, f'{pid}_res{first_interface}_subgraph_{int(top_pct*100)}pct.pdf'),
                            format='pdf', bbox_inches='tight', dpi=150)
                print(f"Saved {pid} subgraph (residue {first_interface}, top-{pct_label})!")
                plt.close()



def test_kfold_models(model_dir, model_fmt, test_data_path, k_folds=5):
    """Evaluate k-fold models on a given test set.

    Model_fmt can be e.g. 'Model_fold{}.pth' (uncalibrated) or
    'Model_fold{}_calibrated.pth' (temperature-scaled models saved by demo.py).
    """
    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    data_loader = PPIData()
    all_proteins = data_loader.load_data(test_data_path)
    val_dataset = PPIDataset(all_proteins)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=sparse_collate,
        pin_memory=True
    )
    print(f'Test dataset size: {len(val_dataset)}')

    for fold in range(1, k_folds+1):
        model_path = os.path.join(model_dir, model_fmt.format(fold))
        print(f'Loading model for fold {fold}: {model_path}')
        model, T = _load_model(model_path)

        explainer = build_explainer(model)
        explain_first_binding_site(model, val_loader, explainer)
        break

if __name__ == '__main__':
    model_dir = '../gz-data/Pre_model/'
    model_fmt = 'Model_fold{}_calibrated.pth'
    k_folds = config.K_FOLDS
    test_data_path = config.VAL2

    test_kfold_models(
        model_dir=model_dir,
        model_fmt=model_fmt,
        test_data_path=test_data_path,
        k_folds=k_folds,
    )
