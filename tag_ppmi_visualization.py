"""PPMI-based visualizations for Codeforces tags.

This script loads ``filtered_problems.csv`` from ``data/problems`` which
contains Codeforces problem metadata.  It assumes **37** unique tags are
present.  For each difficulty group as well as the entire data set the script
produces

1. a co-occurrence matrix and PPMI matrix (min-max scaled)
2. an improved force-directed graph
3. several improved UMAP projections

All outputs are stored under ``data/visualization/<group>``.
"""

from __future__ import annotations

import csv
from collections import Counter
from typing import Dict, List
import os
import logging
import time

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import community
from adjustText import adjust_text
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import umap

# Paths
PROBLEMS_PATH = os.path.join("data", "problems", "filtered_problems.csv")
VIS_DIR = os.path.join("data", "visualization")
os.makedirs(VIS_DIR, exist_ok=True)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


# ------------------------- Helper Functions -------------------------

def parse_tags(tag_str: str) -> List[str]:
    """Convert comma separated tag string to list of tags."""
    if isinstance(tag_str, str):
        return [t.strip() for t in tag_str.split(',') if t.strip()]
    return []


def build_tag_list(tags_series: pd.Series) -> List[str]:
    """Extract sorted unique tags from pandas Series."""
    tag_set = set()
    for tags in tags_series:
        tag_set.update(tags)
    tag_list = sorted(tag_set)
    assert len(tag_list) == 37, "Expected 37 unique tags"
    return tag_list


def build_cooccurrence_matrix(tag_lists: List[List[str]], tag_to_idx: Dict[str, int]) -> np.ndarray:
    """Return co-occurrence matrix (37x37)."""
    size = len(tag_to_idx)
    C = np.zeros((size, size), dtype=int)
    for tags in tag_lists:
        unique_tags = list(dict.fromkeys(tags))
        for i, tag_i in enumerate(unique_tags):
            idx_i = tag_to_idx[tag_i]
            for tag_j in unique_tags[i + 1:]:
                idx_j = tag_to_idx[tag_j]
                C[idx_i, idx_j] += 1
                C[idx_j, idx_i] += 1
    return C


def compute_ppmi_matrix(C: np.ndarray) -> np.ndarray:
    """Compute a min-max scaled PPMI matrix."""
    total = float(C.sum())
    if total == 0:
        return np.zeros_like(C, dtype=float)

    row_sums = C.sum(axis=1).astype(float)
    size = C.shape[0]
    M = np.zeros((size, size), dtype=float)

    # PMI calculation with error handling to avoid log overflow/invalid values
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(size):
            if row_sums[i] == 0:
                continue
            P_i = row_sums[i] / total
            for j in range(size):
                if C[i, j] == 0 or row_sums[j] == 0:
                    continue
                P_j = row_sums[j] / total
                P_ij = C[i, j] / total
                pmi = np.log(P_ij / (P_i * P_j))
                if pmi > 0:
                    M[i, j] = pmi

    # scale the matrix to [0, 1] for downstream weighting
    max_val = M.max()
    if max_val > 0:
        M /= max_val
    return M


def compute_tag_frequency(tag_lists: List[List[str]]) -> Counter:
    """Return Counter of tag frequencies."""
    counter = Counter()
    for tags in tag_lists:
        counter.update(tags)
    return counter


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


def draw_umap_plotly(
    M: np.ndarray,
    tag_list: List[str],
    marker_sizes: Dict[str, float],
    tag_freq: Counter,
    ratios: Dict[str, float],
    deg_cent: Dict[str, float],
    cluster_id: Dict[str, int],
    pair_weights: List[tuple],
    out_dir: str,
) -> None:
    """Interactive UMAP scatter with edge overlay."""

    reducer = umap.UMAP(n_neighbors=10, min_dist=0.2, metric="cosine", random_state=42)
    coords = reducer.fit_transform(M)

    df = pd.DataFrame({
        "tag": tag_list,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "freq": [tag_freq.get(t, 0) for t in tag_list],
        "ratio": [ratios[t] for t in tag_list],
        "marker_size": [marker_sizes[t] for t in tag_list],
        "degree_centrality": [deg_cent[t] for t in tag_list],
        "cluster_id": [cluster_id[t] for t in tag_list],
    })
    df.to_csv(os.path.join(out_dir, "tag_coords_umap.csv"), index=False)

    # prepare edge traces
    sorted_pairs = sorted(pair_weights, key=lambda x: x[2], reverse=True)
    edges_all = sorted_pairs[:100]
    edges_strong = sorted_pairs[:30]

    def make_edge_trace(pairs):
        xs, ys = [], []
        for a, b, _ in pairs:
            p1 = df.loc[df["tag"] == a, ["x", "y"]].values[0]
            p2 = df.loc[df["tag"] == b, ["x", "y"]].values[0]
            xs.extend([p1[0], p2[0], None])
            ys.extend([p1[1], p2[1], None])
        return go.Scattergl(x=xs, y=ys, mode="lines", line=dict(color="gray", width=1), hoverinfo="skip")

    edge_trace_all = make_edge_trace(edges_all)
    edge_trace_strong = make_edge_trace(edges_strong)
    edge_trace_strong.visible = False

    color_seq = px.colors.qualitative.Dark24
    node_colors = [color_seq[cluster_id[t] % len(color_seq)] for t in tag_list]

    node_trace = go.Scattergl(
        x=df["x"],
        y=df["y"],
        mode="markers+text",
        text=df["tag"],
        textposition="top center",
        hovertemplate=(
            "tag: %{text}<br>freq: %{customdata[0]}<br>ratio: %{customdata[1]:.2f}<br>degree_centrality: %{customdata[2]:.3f}<extra></extra>"
        ),
        customdata=df[["freq", "ratio", "degree_centrality"]].values,
        marker=dict(size=df["marker_size"], color=node_colors, line=dict(width=0.5, color="black")),
    )

    fig = go.Figure(data=[edge_trace_all, edge_trace_strong, node_trace])
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=[
                    dict(label="All edges", method="update", args=[{"visible": [True, False, True]}]),
                    dict(label="Strong only", method="update", args=[{"visible": [False, True, True]}]),
                ],
                showactive=True,
                x=0.02,
                y=1.1,
            )
        ],
        title="Interactive UMAP of Tags",
    )

    fig.write_html(os.path.join(out_dir, "umap_interactive.html"))
    fig.write_image(os.path.join(out_dir, "umap_static.png"), scale=2)


def draw_heatmap(
    M: np.ndarray,
    tag_list: List[str],
    tag_freq: Counter,
    cluster_id: Dict[str, int],
    tag_to_idx: Dict[str, int],
    out_dir: str,
) -> None:
    """PPMI heatmap for top 15 tags with cluster color bars."""

    top_tags = [t for t, _ in tag_freq.most_common(15)]

    cluster_tags: Dict[int, List[str]] = {}
    for t in top_tags:
        cluster_tags.setdefault(cluster_id[t], []).append(t)

    ordered_tags: List[str] = []
    for cid in sorted(cluster_tags.keys()):
        tags = cluster_tags[cid]
        tags.sort(key=lambda x: -M[tag_to_idx[x]].sum())
        ordered_tags.extend(tags)

    idxs = [tag_to_idx[t] for t in ordered_tags]
    sub = M[np.ix_(idxs, idxs)]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        sub,
        annot=True,
        fmt=".2f",
        cmap="rocket_r",
        xticklabels=ordered_tags,
        yticklabels=ordered_tags,
        ax=ax,
    )
    ax.set_title("PPMI Heatmap (Top 15 Tags)")

    colors = px.colors.qualitative.Dark24
    for i, tag in enumerate(ordered_tags):
        c = colors[cluster_id[tag] % len(colors)]
        rect_top = plt.Rectangle((i, -0.5), 1, 0.3, facecolor=c, transform=ax.transData, clip_on=False)
        rect_bot = plt.Rectangle((i, len(ordered_tags)-0.5), 1, 0.3, facecolor=c, transform=ax.transData, clip_on=False)
        ax.add_patch(rect_top)
        ax.add_patch(rect_bot)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ppmi_heatmap_top15.png"), dpi=300, bbox_inches="tight")
    plt.close()


def run_for_group(
    tag_lists: List[List[str]],
    tag_list: List[str],
    tag_to_idx: Dict[str, int],
    prefix: str,
    out_root: str,
) -> None:
    """Compute matrices and visualizations for a subset of problems."""
    start_time = time.time()
    logging.info(f"시작: {prefix} 그룹 처리 중...")

    group_dir = os.path.join(out_root, prefix)
    os.makedirs(group_dir, exist_ok=True)

    # ----- Step 1: build co-occurrence and PPMI matrices -----
    logging.info(f"  {prefix}: 공출현 행렬 계산 중...")
    C = build_cooccurrence_matrix(tag_lists, tag_to_idx)

    logging.info(f"  {prefix}: PPMI 행렬 계산 중...")
    M = compute_ppmi_matrix(C)

    # ----- Step 2: tag frequencies and marker size -----
    logging.info(f"  {prefix}: 태그 빈도 계산 중...")
    tag_freq = compute_tag_frequency(tag_lists)
    n_problems = len(tag_lists)
    ratios = {t: (tag_freq.get(t, 0) / n_problems) if n_problems else 0 for t in tag_list}
    marker_sizes = {t: max(ratios[t] * 1000 + 50, 40) for t in tag_list}

    # save basic csv files
    tag_list_df = pd.DataFrame({
        "tag": tag_list,
        "freq": [tag_freq.get(t, 0) for t in tag_list],
        "ratio": [ratios[t] for t in tag_list],
        "marker_size": [marker_sizes[t] for t in tag_list],
    })
    tag_list_df.to_csv(os.path.join(group_dir, "tag_list.csv"), index=False)

    pd.DataFrame(C, index=tag_list, columns=tag_list).to_csv(
        os.path.join(group_dir, "cooccurrence_matrix.csv")
    )
    pd.DataFrame(M, index=tag_list, columns=tag_list).to_csv(
        os.path.join(group_dir, "ppmi_matrix.csv")
    )

    # prepare weighted tag pairs
    triu_i, triu_j = np.triu_indices_from(M, k=1)
    pair_weights = [
        (tag_list[i], tag_list[j], M[i, j])
        for i, j in zip(triu_i, triu_j)
        if M[i, j] > 0
    ]

    # ----- Build graph and Louvain clusters -----
    G = nx.Graph()
    for t_i, t_j, w in pair_weights:
        G.add_edge(t_i, t_j, weight=w)

    deg_cent = nx.degree_centrality(G)
    communities = community.louvain_communities(G, weight="weight", seed=42)
    cluster_id = {n: idx for idx, com in enumerate(communities) for n in com}

    pd.DataFrame(pair_weights, columns=["tag_i", "tag_j", "weight"]).to_csv(
        os.path.join(group_dir, "edges_all.csv"), index=False
    )

    # ----- Visualizations -----
    logging.info(f"  {prefix}: Force-directed 그래프 생성 중...")
    draw_force_directed(G, marker_sizes, deg_cent, cluster_id, group_dir)

    logging.info(f"  {prefix}: UMAP 시각화 생성 중...")
    draw_umap_plotly(
        M,
        tag_list,
        marker_sizes,
        tag_freq,
        ratios,
        deg_cent,
        cluster_id,
        pair_weights,
        group_dir,
    )
    logging.info(f"  {prefix}: 히트맵 생성 중...")
    draw_heatmap(M, tag_list, tag_freq, cluster_id, tag_to_idx, group_dir)

    elapsed = time.time() - start_time
    logging.info(f"완료: {prefix} 그룹 처리 (소요시간: {elapsed:.2f}초)")


# ------------------------- Main Pipeline -------------------------

def main() -> None:
    logging.info("프로그램 시작")
    # 1. CSV 파일 읽기
    df = pd.read_csv(
        PROBLEMS_PATH,
        usecols=[
            'problem_id',
            'name',
            'contestId',
            'index',
            'rating',
            'tags',
            'difficulty_group',
        ],
    )
    df['tags'] = df['tags'].apply(parse_tags)

    # 2. 고유 태그 추출 (총 37개)
    tag_list = build_tag_list(df['tags'])
    tag_to_idx = {tag: idx for idx, tag in enumerate(tag_list)}

    logging.info("전체 데이터 시각화 시작")
    run_for_group(df['tags'].tolist(), tag_list, tag_to_idx, 'All', VIS_DIR)

    logging.info("난이도 그룹별 시각화 시작")
    difficulty_groups = df['difficulty_group'].unique()
    logging.info(f"총 {len(difficulty_groups)}개 난이도 그룹 처리 예정: {difficulty_groups}")

    for i, (group, df_group) in enumerate(df.groupby('difficulty_group')):
        safe_group = str(group).replace(' ', '_').replace('/', '_')
        logging.info(f"난이도 그룹 처리 중 ({i+1}/{len(difficulty_groups)}): {group}")
        run_for_group(
            df_group['tags'].tolist(),
            tag_list,
            tag_to_idx,
            safe_group,
            VIS_DIR,
        )

    logging.info("모든 처리 완료")


if __name__ == '__main__':
    main()
