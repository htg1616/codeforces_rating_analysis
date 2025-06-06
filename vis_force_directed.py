"""Force-directed graph visualization for Codeforces tags."""

from __future__ import annotations

from typing import Dict
import os

import matplotlib.pyplot as plt
import networkx as nx
from adjustText import adjust_text
import logging
import pandas as pd
import numpy as np


def draw_force_directed(
    G: nx.Graph,
    marker_sizes: Dict[str, float],
    deg_cent: Dict[str, float],
    cluster_id: Dict[str, int],
    out_dir: str,
) -> None:
    """Visualize graph using spring layout with Louvain coloring."""

    # 그래프 정보 로깅
    logging.info(f"  Force-Directed: 간선 수: {len(G.edges())}, 고립 노드: {sum(d==0 for _,d in G.degree())}")

    # 노드 간격 확대 (k=0.3 → k=0.5) 및 반복 횟수 감소(속도 향상)
    pos = nx.spring_layout(G, weight="weight", seed=42, k=0.5, iterations=50)

    # 먼저 figure 생성
    plt.figure(figsize=(8, 6))

    node_sizes = [max(marker_sizes[n], 40) for n in G.nodes]
    node_colors = [deg_cent[n] for n in G.nodes]

    # 최대 가중치로 정규화
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()] if G.edges() else [1.0]
    max_w = max(edge_weights) if edge_weights else 1.0

    # 간선 그리기 - 더 가시적으로
    for u, v, w in G.edges(data="weight"):
        # 제곱근 적용으로 얇은 간선도 보이게
        width = np.sqrt(w/max_w) * 5
        same_cluster = cluster_id[u] == cluster_id[v]
        # 0.3 → 0.5로 투명도 조정으로 더 잘 보이게
        alpha = 0.8 if same_cluster else 0.5
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            width=width, alpha=alpha, edge_color="gray"
        )

    # 노드 그리기
    nodes = nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes,
        node_color=node_colors, cmap="cool",
        edgecolors="black", linewidths=0.8
    )

    # 텍스트 라벨 조정
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

    # 중심성 정보 저장 (기존 코드 유지)
    (pd.DataFrame(
        sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:25],
        columns=["tag", "degree_centrality"],
    )).to_csv(os.path.join(out_dir, "centrality_table.csv"), index=False)