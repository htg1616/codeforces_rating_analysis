"""Force-directed graph visualization for Codeforces tags."""

from __future__ import annotations

from typing import Dict, Optional
import os
import logging

import matplotlib.pyplot as plt
import networkx as nx
from adjustText import adjust_text
import pandas as pd
import numpy as np


def draw_force_directed(
    G: nx.Graph,
    marker_sizes: Dict[str, float],
    deg_cent: Dict[str, float],
    cluster_id: Dict[str, int],
    out_dir: str,
    data_dir: Optional[str] = None,
) -> None:
    """Visualize graph using spring layout with Louvain coloring."""

    logging.info(
        f"  Force-Directed: edges={len(G.edges())}, isolated={sum(d==0 for _, d in G.degree())}"
    )

    pos = nx.spring_layout(G, weight="weight", seed=42, k=0.5, iterations=50)

    plt.figure(figsize=(8, 6))

    node_sizes = [max(marker_sizes[n], 40) for n in G.nodes]
    node_colors = [deg_cent[n] for n in G.nodes]

    edge_weights = [G[u][v]["weight"] for u, v in G.edges()] if G.edges() else [1.0]
    max_w = max(edge_weights) if edge_weights else 1.0

    for u, v, w in G.edges(data="weight"):
        width = np.sqrt(w / max_w) * 5
        same_cluster = cluster_id[u] == cluster_id[v]
        alpha = 0.8 if same_cluster else 0.5
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=width,
            alpha=alpha,
            edge_color="gray",
        )

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap="cool",
        edgecolors="black",
        linewidths=0.8,
    )

    texts = [plt.text(x, y, n, fontsize=8) for n, (x, y) in pos.items()]
    adjust_text(texts, arrowprops=dict(arrowstyle="-"))

    plt.colorbar(nodes, label="Degree Centrality")
    plt.title("Force-Directed Tag Graph (Louvain)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "force_directed.png"), dpi=300, bbox_inches="tight")
    plt.close()

    df = pd.DataFrame(
        sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:25],
        columns=["tag", "degree_centrality"],
    )
    if data_dir is not None:
        os.makedirs(data_dir, exist_ok=True)
        df.to_csv(os.path.join(data_dir, "centrality_table.csv"), index=False)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Draw force-directed graph for a difficulty group")
    parser.add_argument("group", help="Difficulty group name, e.g. All or Pupil")
    args = parser.parse_args()

    in_dir = os.path.join("data", args.group)
    out_dir = os.path.join("figures", args.group)
    data_dir = os.path.join("data", args.group)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    tag_df = pd.read_csv(os.path.join(in_dir, "tag_list.csv"))
    tag_list = tag_df["tag"].tolist()
    marker_sizes = dict(zip(tag_df["tag"], tag_df["marker_size"]))
    deg_cent = dict(zip(tag_df["tag"], tag_df["degree_centrality"]))
    cluster_id = dict(zip(tag_df["tag"], tag_df["cluster_id"]))

    edges = pd.read_csv(os.path.join(in_dir, "edges_all.csv"))
    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(row["tag_i"], row["tag_j"], weight=row["weight"])

    draw_force_directed(G, marker_sizes, deg_cent, cluster_id, out_dir, data_dir)


if __name__ == "__main__":
    main()
