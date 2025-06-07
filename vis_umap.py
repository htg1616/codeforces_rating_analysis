"""UMAP scatter visualization for Codeforces tags."""

from __future__ import annotations

from typing import Dict, List
import os
import logging
import sys

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import matplotlib as mpl
from adjustText import adjust_text

# 한글 폰트 설정 추가
import platform

# 운영체제별 기본 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')  # 윈도우의 경우 '맑은 고딕' 폰트 사용
elif platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')  # macOS의 경우 'AppleGothic' 사용
else:  # Linux 등 기타 운영체제
    # 나눔글꼴이나 다른 한�� 폰트가 설치되어 있다면 사용
    try:
        plt.rc('font', family='NanumGothic')
    except:
        pass  # 설치된 한글 폰트가 없으면 기본값 사용

# 음수 표시 문제 해결
mpl.rcParams['axes.unicode_minus'] = False

# plotly는 HTML 생성에만 사용 (정적 이미지는 matplotlib로만 생성)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


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

    try:
        # 1. UMAP 차원 축소 - 더 간단한 파라미터로
        logging.info("  UMAP: 차원 축소 시작...")
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
            verbose=False,  # 상세 로그 끄기
        )
        coords = reducer.fit_transform(M)
        logging.info("  UMAP: 차원 축소 완료")

        # 2. 결과 저장용 데이터프레임 생성
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

        # 3. matplotlib 정적 이미지 생성 - 항상 실행됨
        logging.info("  UMAP: matplotlib 정적 이미지 생성 중...")
        create_matplotlib_static(df, pair_weights, cluster_id, out_dir)

        # 4. (선택적) Plotly HTML 생성
        if PLOTLY_AVAILABLE:
            try:
                logging.info("  UMAP: Plotly HTML 생성 중...")
                create_plotly_html(df, pair_weights, cluster_id, out_dir)
            except Exception as e:
                logging.error(f"  UMAP: Plotly HTML 생성 실패: {str(e)}")

    except Exception as e:
        logging.error(f"  UMAP: 예상치 못한 오류 발생: {str(e)}")
        # 오류가 발생해도 프로그램 계속 실행


def create_matplotlib_static(df, pair_weights, cluster_id, out_dir):
    """안정적인 matplotlib 정적 이미지 생성"""
    plt.figure(figsize=(12, 10))

    # 1. 색상 설정 (plotly와 유사한 색상)
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
    cluster_colors = {cid: colors[cid % len(colors)] for cid in set(cluster_id.values())}

    # 2. 간선 그리기 (상위 80개만)
    sorted_pairs = sorted(pair_weights, key=lambda x: x[2], reverse=True)
    for a, b, w in sorted_pairs[:80]:
        try:
            p1 = df.loc[df["tag"] == a, ["x", "y"]].values[0]
            p2 = df.loc[df["tag"] == b, ["x", "y"]].values[0]
            # 가중치에 따라 선 두께와 투명도 조절
            line_alpha = 0.2 + 0.4 * (w / sorted_pairs[0][2])
            line_width = 0.5 + 1.0 * (w / sorted_pairs[0][2])  # 가중치에 따라 선 두께 변화
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', alpha=line_alpha, linewidth=line_width, zorder=1)
        except IndexError:
            continue

    # 3. 노드 그리기 (zorder=5로 간선 위에 표시)
    for i, row in df.iterrows():
        tag = row["tag"]
        cluster = cluster_id.get(tag, 0)
        color = cluster_colors.get(cluster, "#333333")
        size = row["marker_size"] * 0.8
        plt.scatter(row["x"], row["y"], s=size, color=color, edgecolors='black', linewidths=0.5, alpha=0.9, zorder=5)

    # 4. 텍스트 라벨 (겹치지 않게 조정)
    texts = []
    for i, row in df.iterrows():
        # 텍스트 크기를 9에서 8로 줄이고, zorder를 10으로 설정
        # 배경색 추가로 텍스트 가독성 향상
        texts.append(plt.text(
            row["x"], row["y"],
            row["tag"],
            fontsize=8,
            ha='center',
            va='center',
            zorder=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
        ))

    # force_directed와 유사하게 텍스트 배치 방식 단순화
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color="gray", alpha=0.6, lw=0.5),
        expand_text=(1.1, 1.1),  # 텍스트 간 간격 확대
        expand_points=(1.2, 1.2)  # 포인트와 텍스트 간 간격 확대
    )

    plt.title("UMAP 태그 시각화")
    plt.axis('off')
    plt.tight_layout()

    # 이미지 저장
    static_path = os.path.join(out_dir, "umap_static.png")
    plt.savefig(static_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"  UMAP: matplotlib 정적 이미지 저장 완료: {static_path}")


def create_plotly_html(df, pair_weights, cluster_id, out_dir):
    """Plotly 인터랙티브 HTML 생성 (정적 이미지 생성 없음)"""
    # 1. 간선 데이터 준비
    sorted_pairs = sorted(pair_weights, key=lambda x: x[2], reverse=True)
    edges_all = sorted_pairs[:120]
    edges_strong = sorted_pairs[:40]

    # 2. 간선 트레이스 생성
    xs_all, ys_all = [], []
    for a, b, w in edges_all:
        try:
            # 양쪽 태그가 모두 존재하는지 먼저 확인
            if not df[df["tag"] == a].empty and not df[df["tag"] == b].empty:
                p1 = df.loc[df["tag"] == a, ["x", "y"]].values[0]
                p2 = df.loc[df["tag"] == b, ["x", "y"]].values[0]
                xs_all.extend([p1[0], p2[0], None])
                ys_all.extend([p1[1], p2[1], None])
        except IndexError:
            continue

    xs_strong, ys_strong = [], []
    for a, b, w in edges_strong:
        try:
            if not df[df["tag"] == a].empty and not df[df["tag"] == b].empty:
                p1 = df.loc[df["tag"] == a, ["x", "y"]].values[0]
                p2 = df.loc[df["tag"] == b, ["x", "y"]].values[0]
                xs_strong.extend([p1[0], p2[0], None])
                ys_strong.extend([p1[1], p2[1], None])
        except IndexError:
            continue

    edge_trace_all = go.Scatter(
        x=xs_all, y=ys_all,
        mode="lines",
        line=dict(color="rgba(180,180,180,0.3)", width=1),
        hoverinfo="none"
    )
    # 'layer' 속성 제거 - Scatter 객체에서는 지원하지 않음
    # 간선은 추가 순서에 따라 자동으로 노드 아래에 배치됨

    edge_trace_strong = go.Scatter(
        x=xs_strong, y=ys_strong,
        mode="lines",
        line=dict(color="rgba(100,100,100,0.7)", width=1.5),
        hoverinfo="none",
        visible=False
    )
    # 'layer' 속성 제거

    # 3. 노드 트레이스
    color_seq = px.colors.qualitative.Plotly
    node_colors = [color_seq[cluster_id[t] % len(color_seq)] for t in df["tag"]]

    # ------------------------------------------
    # 1) 반지름 → 면적 변환 (크기 대폭 축소)
    # ------------------------------------------
    desired_max_r = 6           # 화면에 보일 최대 반지름(px) - 12에서 6으로 축소
    radius_series = df["marker_size"] * 0.6  # 원본 반지름 값을 0.6배로 축소
    area_series = radius_series ** 2     # 면적으로 변환
    df["plot_size"] = area_series

    # ------------------------------------------
    # 2) sizeref 계산 - 정확한 공식 적용
    # ------------------------------------------
    max_area = area_series.max()
    sizeref = 2.0 * max_area / (desired_max_r ** 2)

    node_trace = go.Scatter(
        x=df["x"],
        y=df["y"],
        mode="markers+text",
        text=df["tag"],
        textposition="top center",
        textfont=dict(size=10, color="black"),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "빈도: %{customdata[0]}<br>"
            "비율: %{customdata[1]:.2f}<br>"
            "중심성: %{customdata[2]:.3f}"
            "<extra></extra>"
        ),
        customdata=df[["freq", "ratio", "degree_centrality"]].values,
        marker=dict(
            size=df["plot_size"],   # 면적 값 사용
            sizemode="area",        # 반드시 'area'로 지정
            sizeref=sizeref,        # 위에서 계산한 정확한 sizeref 사용
            sizemin=3,              # 최소 크기도 더 작게 (4에서 3으로)
            color=node_colors,
            line=dict(width=0.5, color="black"),  # 테두리도 얇게
            opacity=0.9
        ),
    )

    # 4. 클러스터 Convex Hull 생성 (선택적)
    hull_traces = []
    # 클러스터별로 Convex Hull 생성 (필요한 경우 활성화)
    clusters = df.groupby(df["tag"].map(cluster_id))

    for cid, cluster_df in clusters:
        if len(cluster_df) >= 3:  # 최소 3개 이상의 점이 있어야 Convex Hull 생성 가능
            try:
                from scipy.spatial import ConvexHull
                points = cluster_df[["x", "y"]].values
                hull = ConvexHull(points)
                hull_x = points[hull.vertices, 0].tolist() + [points[hull.vertices[0], 0]]
                hull_y = points[hull.vertices, 1].tolist() + [points[hull.vertices[0], 1]]

                # 색상 처리 간소화 및 가시성 향상
                base_color = color_seq[cid % len(color_seq)]

                # RGB 변환 (간소화된 방법)
                r = int(base_color[1:3], 16)
                g = int(base_color[3:5], 16)
                b = int(base_color[5:7], 16)

                # 클러스터 영역 트레이스 생성
                hull_trace = go.Scatter(
                    x=hull_x, y=hull_y,
                    mode="lines",
                    fill="toself",  # None에서 toself로 변경 - 영역 채우기 활성화
                    fillcolor=f"rgba({r},{g},{b},0.2)",  # 투명도 증가 (0.08에서 0.2로)
                    line=dict(width=1.5, color=f"rgba({r},{g},{b},0.6)"),  # 테두리도 더 진하게
                    showlegend=True,
                    name=f"Cluster {cid}",
                    legendgroup=f"cluster{cid}",
                    hoverinfo="skip",
                    visible="legendonly"  # 기본적으로 숨김 상태
                )
                hull_traces.append(hull_trace)
            except (ImportError, ValueError):
                pass  # ConvexHull 생성 실패 시 무시

    # 5. 그래프 생성 및 레이아웃 설정
    # 트레이스 추가 순서 조정: 간선을 먼저, 노드를 나중에 추가하여 노드가 간선 위에 표시되게 함
    all_traces = hull_traces + [edge_trace_all, edge_trace_strong, node_trace]
    fig = go.Figure(data=all_traces)

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            buttons=[
                dict(label="모든 간선", method="update", args=[{"visible": [True if i < len(hull_traces) else [True, False, True][i-len(hull_traces)] for i in range(len(all_traces))]}]),
                dict(label="강한 간선만", method="update", args=[{"visible": [True if i < len(hull_traces) else [False, True, True][i-len(hull_traces)] for i in range(len(all_traces))]}]),
            ],
            showactive=True,
            x=0.02,
            y=1.1,
        )],
        title="태그 UMAP 시각화",
        plot_bgcolor="white",
        width=900,
        height=700,
        margin=dict(t=50, b=50, l=50, r=50),
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgba(200,200,200,0.2)', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(200,200,200,0.2)', zeroline=False)

    # 6. HTML 파일만 저장
    html_path = os.path.join(out_dir, "umap_interactive.html")
    fig.write_html(html_path)
    logging.info(f"  UMAP: HTML 파일 저장 완료: {html_path}")



def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Draw UMAP visualization for a difficulty group")
    parser.add_argument("group", help="Difficulty group name, e.g. All or Pupil")
    args = parser.parse_args()

    in_dir = os.path.join("data", args.group)
    out_dir = os.path.join("figures", args.group)
    os.makedirs(out_dir, exist_ok=True)

    tag_df = pd.read_csv(os.path.join(in_dir, "tag_list.csv"))
    tag_list = tag_df["tag"].tolist()
    marker_sizes = dict(zip(tag_df["tag"], tag_df["marker_size"]))
    tag_freq = dict(zip(tag_df["tag"], tag_df["freq"]))
    ratios = dict(zip(tag_df["tag"], tag_df["ratio"]))
    deg_cent = dict(zip(tag_df["tag"], tag_df["degree_centrality"]))
    cluster_id = dict(zip(tag_df["tag"], tag_df["cluster_id"]))

    M = pd.read_csv(os.path.join(in_dir, "ppmi_matrix.csv"), index_col=0).values
    pair_weights = pd.read_csv(os.path.join(in_dir, "edges_all.csv")).values.tolist()

    draw_umap_plotly(
        M,
        tag_list,
        marker_sizes,
        tag_freq,
        ratios,
        deg_cent,
        cluster_id,
        pair_weights,
        out_dir,
    )


if __name__ == "__main__":
    main()
