# -*- coding: utf-8 -*-
"""PPMI-based visualizations for Codeforces tags.

This script loads `filtered_problems.csv` which contains Codeforces problem
metadata. It assumes there are exactly 37 unique tags. It produces a
co-occurrence matrix, a PPMI matrix, and two visualizations:
1. UMAP projection
2. Force-directed graph

Output files (per difficulty group):
- tag_list_<group>.csv
- cooccurrence_matrix_<group>.csv
- ppmi_matrix_<group>.csv
- tag_coords_umap_<group>.csv
- umap_tags_plot_<group>.png
- force_directed_tags_plot_<group>.png
- edges_topk_<group>.csv (optional)
"""

from __future__ import annotations

import csv
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import umap


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
    """Compute PPMI matrix from co-occurrence matrix."""
    total = float(C.sum())
    if total == 0:
        return np.zeros_like(C, dtype=float)

    row_sums = C.sum(axis=1).astype(float)
    size = C.shape[0]
    M = np.zeros((size, size), dtype=float)

    with np.errstate(divide='ignore', invalid='ignore'):
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
    return M


def compute_tag_frequency(tag_lists: List[List[str]]) -> Counter:
    """Return Counter of tag frequencies."""
    counter = Counter()
    for tags in tag_lists:
        counter.update(tags)
    return counter


def run_for_group(tag_lists: List[List[str]], tag_list: List[str], tag_to_idx: Dict[str, int], prefix: str) -> None:
    """Compute matrices and visualizations for a subset of problems."""
    # 3. 공출현 행렬 생성
    C = build_cooccurrence_matrix(tag_lists, tag_to_idx)

    # 4. PPMI 행렬 계산
    M = compute_ppmi_matrix(C)

    # 5. 태그별 등장 빈도 계산
    tag_freq = compute_tag_frequency(tag_lists)
    n_problems = len(tag_lists)
    ratios = {tag: (tag_freq.get(tag, 0) / n_problems) if n_problems else 0 for tag in tag_list}
    marker_sizes = {tag: ratios[tag] * 1000 + 50 for tag in tag_list}

    # Save tag_list with frequencies
    tag_list_df = pd.DataFrame({
        'tag': tag_list,
        'freq': [tag_freq.get(t, 0) for t in tag_list],
        'ratio': [ratios[t] for t in tag_list],
        'marker_size': [marker_sizes[t] for t in tag_list],
    })
    tag_list_df.to_csv(f'tag_list_{prefix}.csv', index=False)

    # Save co-occurrence matrix
    coocc_df = pd.DataFrame(C, index=tag_list, columns=tag_list)
    coocc_df.to_csv(f'cooccurrence_matrix_{prefix}.csv')

    # Save PPMI matrix
    ppmi_df = pd.DataFrame(M, index=tag_list, columns=tag_list)
    ppmi_df.to_csv(f'ppmi_matrix_{prefix}.csv')

    # ----------------- Visualization 1: UMAP -----------------
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.05, n_components=2, random_state=42)
    tag_coords_umap = reducer.fit_transform(M)

    coords_df = pd.DataFrame({
        'tag': tag_list,
        'x': tag_coords_umap[:, 0],
        'y': tag_coords_umap[:, 1],
        'freq': [tag_freq.get(t, 0) for t in tag_list],
        'ratio': [ratios[t] for t in tag_list],
        'marker_size': [marker_sizes[t] for t in tag_list],
    })
    coords_df.to_csv(f'tag_coords_umap_{prefix}.csv', index=False)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        coords_df['x'],
        coords_df['y'],
        s=coords_df['marker_size'],
        c=coords_df['ratio'],
        cmap='viridis',
        alpha=0.8,
        edgecolors='k',
    )
    for _, row in coords_df.iterrows():
        plt.text(row['x'], row['y'], row['tag'], fontsize=8, ha='center', va='center')
    plt.title('UMAP Projection of 37 Codeforces Tags (PPMI-based)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Frequency Ratio')
    plt.tight_layout()
    plt.savefig(f'umap_tags_plot_{prefix}.png', dpi=300)
    plt.close()

    # ----------------- Visualization 2: Force-Directed Graph -----------------
    G = nx.Graph()
    G.add_nodes_from(tag_list)

    triu_idx = np.triu_indices_from(M, k=1)
    weights = M[triu_idx]
    threshold = np.percentile(weights[weights > 0], 90) if np.any(weights > 0) else 0

    edges = []
    for i, j in zip(*triu_idx):
        weight = M[i, j]
        if weight > threshold:
            edges.append((tag_list[i], tag_list[j], weight))
            G.add_edge(tag_list[i], tag_list[j], weight=weight)

    if edges:
        edge_df = pd.DataFrame(edges, columns=['tag_i', 'tag_j', 'weight'])
        edge_df.to_csv(f'edges_topk_{prefix}.csv', index=False)

    pos = nx.spring_layout(G, weight='weight', seed=42, k=0.5)

    plt.figure(figsize=(8, 6))
    node_sizes = [marker_sizes[tag] for tag in G.nodes]
    edge_weights = [G[u][v]['weight'] for u, v in G.edges]
    if edge_weights:
        max_w = max(edge_weights)
        edge_widths = [w * 5 / max_w for w in edge_weights]
    else:
        edge_widths = []

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', edgecolors='black')
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title('Force-Directed Layout of 37 Codeforces Tags (PPMI-based)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'force_directed_tags_plot_{prefix}.png', dpi=300)
    plt.close()


# ------------------------- Main Pipeline -------------------------

def main() -> None:
    # 1. CSV 파일 읽기
    df = pd.read_csv(
        'filtered_problems.csv',
        usecols=['problem_id', 'name', 'contestId', 'index', 'rating', 'tags', 'difficulty_group'],
    )
    df['tags'] = df['tags'].apply(parse_tags)

    # 2. 고유 태그 추출 (총 37개)
    tag_list = build_tag_list(df['tags'])
    tag_to_idx = {tag: idx for idx, tag in enumerate(tag_list)}

    # 3. 난이도 그룹별 시각화
    for group, df_group in df.groupby('difficulty_group'):
        safe_group = str(group).replace(' ', '_').replace('/', '_')
        run_for_group(df_group['tags'].tolist(), tag_list, tag_to_idx, safe_group)


if __name__ == '__main__':
    main()
