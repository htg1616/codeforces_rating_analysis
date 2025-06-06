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

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import umap

# Paths
PROBLEMS_PATH = os.path.join("data", "problems", "filtered_problems.csv")
VIS_DIR = os.path.join("data", "visualization")
os.makedirs(VIS_DIR, exist_ok=True)


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


def run_for_group(
    tag_lists: List[List[str]],
    tag_list: List[str],
    tag_to_idx: Dict[str, int],
    prefix: str,
    out_root: str,
) -> None:
    """Compute matrices and visualizations for a subset of problems."""
    group_dir = os.path.join(out_root, prefix)
    os.makedirs(group_dir, exist_ok=True)

    # ----- Step 1: build co-occurrence and PPMI matrices -----
    C = build_cooccurrence_matrix(tag_lists, tag_to_idx)
    M = compute_ppmi_matrix(C)

    # ----- Step 2: tag frequencies and marker size -----
    tag_freq = compute_tag_frequency(tag_lists)
    n_problems = len(tag_lists)
    ratios = {t: (tag_freq.get(t, 0) / n_problems) if n_problems else 0 for t in tag_list}
    marker_sizes = {t: ratios[t] * 1000 + 50 for t in tag_list}

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

    # prepare a list of weighted pairs for later use
    triu_i, triu_j = np.triu_indices_from(M, k=1)
    pair_weights = [
        (tag_list[i], tag_list[j], M[i, j]) for i, j in zip(triu_i, triu_j) if M[i, j] > 0
    ]

    # ========================= Improved UMAP =========================
    # top 50 edges by weight for overlay
    top_pairs = sorted(pair_weights, key=lambda x: x[2], reverse=True)[:50]
    for nn in (5, 10, 15):
        reducer = umap.UMAP(
            n_neighbors=nn, min_dist=0.2, metric="cosine", random_state=42
        )
        coords = reducer.fit_transform(M)
        coords_df = pd.DataFrame({
            "tag": tag_list,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "freq": [tag_freq.get(t, 0) for t in tag_list],
            "ratio": [ratios[t] for t in tag_list],
            "marker_size": [marker_sizes[t] for t in tag_list],
        })
        coords_df.to_csv(
            os.path.join(group_dir, f"tag_coords_umap_nn{nn}.csv"), index=False
        )

        plt.figure(figsize=(8, 6))
        # overlay thin gray edges for strong pairs
        for t_i, t_j, _ in top_pairs:
            p1 = coords_df.loc[coords_df["tag"] == t_i, ["x", "y"]].values[0]
            p2 = coords_df.loc[coords_df["tag"] == t_j, ["x", "y"]].values[0]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", alpha=0.3, linewidth=0.5)

        scatter = plt.scatter(
            coords_df["x"],
            coords_df["y"],
            s=coords_df["marker_size"],
            c=coords_df["ratio"],
            cmap="viridis",
            alpha=0.9,
            edgecolors="k",
        )
        for _, row in coords_df.iterrows():
            plt.text(row["x"], row["y"], row["tag"], fontsize=8, ha="center", va="center")

        plt.colorbar(scatter, label="Frequency Ratio")
        plt.title(
            f"UMAP (n_neighbors={nn}, min_dist=0.2, metric='cosine')"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(group_dir, f"umap_tags_nn{nn}.png"), dpi=300)
        plt.close()

    # ====================== Improved Force-Directed ======================
    G = nx.Graph()
    G.add_nodes_from(tag_list)

    weights = np.array([w for _, _, w in pair_weights])
    threshold = np.percentile(weights, 70) if weights.size else 0

    edges = []
    for t_i, t_j, w in pair_weights:
        if w > threshold:
            G.add_edge(t_i, t_j, weight=w)
            edges.append((t_i, t_j, w))

    pd.DataFrame(edges, columns=["tag_i", "tag_j", "weight"]).to_csv(
        os.path.join(group_dir, "edges_force_directed.csv"), index=False
    )

    pos = nx.spring_layout(G, weight="weight", seed=42, k=1)
    deg_cent = nx.degree_centrality(G)

    node_sizes = [marker_sizes[n] for n in G.nodes]
    node_colors = [deg_cent[n] for n in G.nodes]
    edge_weights = [G[u][v]["weight"] for u, v in G.edges]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [np.sqrt(w / max_w) * 4 for w in edge_weights]

    plt.figure(figsize=(8, 6))
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap="cool",
        edgecolors="black",
    )
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.colorbar(nodes, label="Degree Centrality")
    plt.title("Force-Directed Layout of 37 Codeforces Tags (PPMI, thresh=70%)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(group_dir, "force_directed_tags_improved.png"), dpi=300
    )
    plt.close()


# ------------------------- Main Pipeline -------------------------

def main() -> None:
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

    # 3. 전체 데이터 시각화
    run_for_group(df['tags'].tolist(), tag_list, tag_to_idx, 'All', VIS_DIR)

    # 4. 난이도 그룹별 시각화
    for group, df_group in df.groupby('difficulty_group'):
        safe_group = str(group).replace(' ', '_').replace('/', '_')
        run_for_group(
            df_group['tags'].tolist(),
            tag_list,
            tag_to_idx,
            safe_group,
            VIS_DIR,
        )


if __name__ == '__main__':
    main()
