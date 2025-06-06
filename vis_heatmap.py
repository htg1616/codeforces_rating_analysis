"""Heatmap visualization for Codeforces tag PPMI."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List
import os

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import pandas as pd


def draw_heatmap(
    M: np.ndarray,
    tag_list: List[str],
    tag_freq: Dict[str, int],
    cluster_id: Dict[str, int],
    tag_to_idx: Dict[str, int],
    out_dir: str,
) -> None:
    """PPMI heatmap for top 15 tags with cluster color bars."""

    top_tags = [t for t, _ in sorted(tag_freq.items(), key=lambda x: -x[1])[:15]]

    cluster_tags: Dict[int, List[str]] = defaultdict(list)
    for t in top_tags:
        cluster_tags[cluster_id[t]].append(t)

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
        rect_bot = plt.Rectangle((i, len(ordered_tags) - 0.5), 1, 0.3, facecolor=c, transform=ax.transData, clip_on=False)
        ax.add_patch(rect_top)
        ax.add_patch(rect_bot)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ppmi_heatmap_top15.png"), dpi=300, bbox_inches="tight")
    plt.close()

