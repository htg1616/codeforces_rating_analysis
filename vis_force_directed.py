"""Force-directed graph visualization for Codeforces tags."""

from __future__ import annotations

from typing import Dict
import os

import matplotlib.pyplot as plt
import networkx as nx
from adjustText import adjust_text
import pandas as pd


def draw_force_directed(
    G: nx.Graph,
    marker_sizes: Dict[str, float],
    deg_cent: Dict[str, float],
    cluster_id: Dict[str, int],
    out_dir: str,
) -> None:
    """Visualize graph using spring layout with Louvain coloring."""

    pos = nx.spring_layout(G, weight="weight", seed=42)

    node_sizes = [max(marker_sizes[n], 40) for n in G.nodes]
    node_colors = [deg_cent[n] for n in G.nodes]

    for u, v, w in G.edges(data="weight"):
        width = np.sqrt(w) * 4
        same_cluster = cluster_id[u] == cluster_id[v]
        alpha = 0.80 if same_cluster else 0.30
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=width,
            alpha=alpha,
            edge_color="gray",
        )

    plt.figure(figsize=(8, 6))
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap="cool",
        edgecolors="black",
        linewidths=0.8,
    )

    texts = []
    for n, (x, y) in pos.items():
        texts.append(plt.text(x, y, n, fontsize=8))
    adjust_text(texts, arrowprops=dict(arrowstyle="-"))

    plt.colorbar(nodes, label="Degree Centrality")
    plt.title("Force-Directed Tag Graph (Louvain)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "force_all_louvain.png"), dpi=300, bbox_inches="tight")
    plt.close()

    (
        pd.DataFrame(
            sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:25],
            columns=["tag", "degree_centrality"],
        )
    ).to_csv(os.path.join(out_dir, "centrality_table.csv"), index=False)

