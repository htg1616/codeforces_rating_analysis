import logging
import os
from typing import List, Dict

import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import community

from data_processing import (
    parse_tags,
    build_tag_list,
    build_cooccurrence_matrix,
    compute_ppmi_matrix,
    compute_tag_frequency,
)

PROBLEMS_PATH = os.path.join("data", "problems", "filtered_problems.csv")
DATA_DIR = "data"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def process_group(tag_lists: List[List[str]], tag_list: List[str], tag_to_idx: Dict[str, int], prefix: str) -> None:
    """Compute matrices and statistics for a subset of problems."""
    out_dir = os.path.join(DATA_DIR, prefix)
    os.makedirs(out_dir, exist_ok=True)

    logging.info(f"Processing group: {prefix}")
    C = build_cooccurrence_matrix(tag_lists, tag_to_idx)
    M = compute_ppmi_matrix(C)

    tag_freq = compute_tag_frequency(tag_lists)
    n = len(tag_lists)
    ratios = {t: (tag_freq.get(t, 0) / n) if n else 0 for t in tag_list}
    marker_sizes = {t: max(ratios[t] * 1000 + 50, 40) for t in tag_list}

    pd.DataFrame(C, index=tag_list, columns=tag_list).to_csv(
        os.path.join(out_dir, "cooccurrence_matrix.csv")
    )
    pd.DataFrame(M, index=tag_list, columns=tag_list).to_csv(
        os.path.join(out_dir, "ppmi_matrix.csv")
    )

    flat = M[np.triu_indices_from(M, k=1)]
    top_k = 120
    idx_k = np.argpartition(flat, -top_k)[-top_k:] if len(flat) > top_k else np.arange(len(flat))
    thr = flat[idx_k].min() if len(flat) > 0 else 0

    pair_weights = [
        (tag_list[i], tag_list[j], M[i, j])
        for i in range(len(tag_list))
        for j in range(i + 1, len(tag_list))
        if M[i, j] >= thr and M[i, j] > 0
    ]
    pd.DataFrame(pair_weights, columns=["tag_i", "tag_j", "weight"]).to_csv(
        os.path.join(out_dir, "edges_all.csv"), index=False
    )

    G = nx.Graph()
    for t_i, t_j, w in pair_weights:
        G.add_edge(t_i, t_j, weight=w)

    deg_cent = nx.degree_centrality(G)
    communities = community.louvain_communities(G, weight="weight", seed=42)
    cluster_id = {n: idx for idx, com in enumerate(communities) for n in com}

    pd.DataFrame({
        "tag": tag_list,
        "freq": [tag_freq.get(t, 0) for t in tag_list],
        "ratio": [ratios[t] for t in tag_list],
        "marker_size": [marker_sizes[t] for t in tag_list],
        "degree_centrality": [deg_cent.get(t, 0) for t in tag_list],
        "cluster_id": [cluster_id.get(t, -1) for t in tag_list],
    }).to_csv(os.path.join(out_dir, "tag_list.csv"), index=False)



def main() -> None:
    df = pd.read_csv(
        PROBLEMS_PATH,
        usecols=[
            "problem_id",
            "name",
            "contestId",
            "index",
            "rating",
            "tags",
            "difficulty_group",
        ],
    )
    df["tags"] = df["tags"].apply(parse_tags)

    tag_list = build_tag_list(df["tags"])
    tag_to_idx = {tag: idx for idx, tag in enumerate(tag_list)}

    process_group(df["tags"].tolist(), tag_list, tag_to_idx, "All")

    for group, df_group in df.groupby("difficulty_group"):
        safe_group = str(group).replace(" ", "_").replace("/", "_")
        process_group(df_group["tags"].tolist(), tag_list, tag_to_idx, safe_group)


if __name__ == "__main__":
    main()
