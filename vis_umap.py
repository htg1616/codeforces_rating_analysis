"""UMAP scatter visualization for Codeforces tags."""

from __future__ import annotations

from typing import Dict, List
import os

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import umap


def draw_umap_plotly(
    M: np.ndarray,
    tag_list: List[str],
    marker_sizes: Dict[str, float],
    tag_freq: Dict[str, int],
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

